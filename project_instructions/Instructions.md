# Instructions

Hello, and thanks for taking the time to interview with us at Tracksuit! This is a take-home exercise we'd like you to complete. This is an open-ended problem that is designed to showcase your thoughtfulness and creativity. To avoid spending too much time on this task, we encourage you to timebox your work to a few hours.

<!-- deno-fmt-ignore-start -->
> [!Note]
> This task is meant to give us something to talk over in your technical
> interview. It is a challenging but realistic example of the kind of
> problems you may tackle at Tracksuit and you're encouraged to use AI
> to assist you. Finally, remember: Nothing is designed to trick you.
> If you have any questions, please reach out! 
<!-- deno-fmt-ignore-end -->

## Background

Surveys are the heart of Tracksuit's operation. The way they are designed affects our ability to develop new products, build trust with our customers, and improve our cash flow. As Tracksuit continues to scale, automated optimisation becomes an increasingly important contributor to our business.

One of the core functions of the survey team is to ensure that every customer gets the sample size they need, while minimising surveying costs and respecting the attention span of survey respondents. 

This is tricky for three reasons:

1. **Every customer has a different category with a different incidence (qualifying) rate.**

At Tracksuit, we guarantee each customer n=200 respondents that qualify for their category each month. When different categories have different incidence rates, we need to ask the category’s qualifying question to different numbers of respondents to get the same final result. 
For example, if the Honey category has a 20% incidence rate and the Chips category has a 80% incidence rate, we’ll need to ask roughly 200/20%=1000 respondents the qualifying question for Honey, while we only need to ask roughly 200/80%=250 respondents the qualifying question for Chips.

2. **Every customer’s survey length is different.**

While we ensure that every customer starts with the same standard set of questions, we allow customers the ability to add questions to their category to measure their unique brand goals. This means that, while basic categories might only last 45 seconds, a fully-fledge survey could take up to 3 minutes to complete.

3. **The mean respondent can only have 8 minutes of questions.**

To ensure that respondents are engaged for the length of the survey, we restrict the total survey length to 8 minutes per respondent. In practice, this means you'll need to design survey structures (or multiple survey versions) that account for the probabilistic nature of qualification rates. Your solution should determine, before the month begins, how to structure surveys to achieve the desired outcomes. 

The demographic distribution of the sample that is exposed to each category’s qualifier should be representative of the national population.
While the demographic composition of those who qualify for each category will naturally vary (for example, the sample of those who qualify for “Men’s Clothing” will naturally skew more male than those for “Women’s Clothing), we want to make sure that those who had the chance to qualify are nationally representative. 

In this task, imagine you’re a Data Scientist at Tracksuit designing an algorithmic solution to this problem.

## The Task

The goal of this take-home task is to minimise the total number of respondents surveyed (our "cost") by designing, implementing, and validating a policy or algorithm that determines how to structure surveys before the month begins. This should define how categories are allocated to create survey structures that probabilistically achieve the desired outcomes while respecting the following constraints: 
- Every category should receive roughly 200 qualified respondents (we're modelling one month). In contracts, we specify that customers will receive at least 2,400 qualified respondents per year.
- The mean respondents should have a total interview length of less than 480 seconds (8 minutes). This limit exists for two reasons: 1) to maintain data quality - we believe that respondent engagement starts to drop rapidly after an 8 minute survey - and, 2) because it's stipulated in our contractual agreement with our sample provider.
*For simplicity, you may assume that the category qualifier consumes none of a respondent's time (0 seconds).However, note that in real-world implementation, there is a practical limit to how many qualifiers can be shown to a single respondent due to respondent burden and survey flow considerations. While your solution may leverage the 0-second assumption, strong candidates will consider how their approach would translate to real-world constraints.*
- The demographic distribution of the sample exposed to each category's qualifiers should match that of the national population (At Tracksuit, we quota for at least the gender, age, and region variables). We require this to avoid respondent bias and maintain best-in-class research practice.

Since we'll be reviewing your work asynchronously, please follow the instructions in the [README](./README.md) to submit your task. You're welcome to use your programming language and framework of choice but please ensure your full solution is reproducible from your code. 

You're also more than welcome to use any AI tools (e.g. Cursor or Claude Code) to help you with this take home task, similarly to how you would as an employee at Tracksuit. However, similarly to any code you write at Tracksuit, you will be responsible for the quality of your code and your results. Make sure you can interpret and explain any code, assumptions, and results that you submit. Please note that we will be comparing your results to a naive Claude Code solution based on this repository. You'll need to contributie additional insight or intuition to achieve a strong performance. 

Validation is a critical part of this feature. As part of your submission, you should provide clear proof that your algorithm achieves each of the constraints listed above, using at least the `fake_category_data.csv` file.
In particular, your validation should include a simulation that demonstrates how your pre-computed policy would perform over the course of a month, showing that it probabilistically meets all requirements.

If you find that a full solution takes you more time than you have available, you may choose to add simplifying assumptions or relax set constraints to assist in your solution, as long as you justify your decision from both a customer and technical perspective in your presentation. This, too, will help us understand how you think about product scope and technical trade-offs.

To help you with this task, we have provided the following description of the data generating process.

### Understanding Categories

Tracksuit categories are defined by a category qualifier. This is a question that specifies the purchase behavior required to "qualify" for the category. For example, the `Fast Food` category has the following category qualifier: "In the past 3 months have you purchased fast-food? (Note: this includes quick and convenient options such as sandwiches/wraps, pizza, burritos, salads, and burgers, etc)"

Each category qualifier also comes with a set of answer options. For the `Fast Food` category above, the answer options are:
- "Yes" (`Qualifying`), and 
- "No"

The `Qualifying` label indicates that survey respondents who select "Yes" qualify for the category.

By deploying the category qualifier with its associated answer options to a survey, we can measure the category's incidence rate. The incidence rate is the estimated percentage of the population that qualifies (selects a `Qualifying` response) to a given category qualifier. In the example above, we surveyed 8,952 people and 7,583 selected "Yes", leading to an estimated incidence rate of `7583/8952=84.71%`. 

After a respondent "qualifies" for a category, they are then eligible to respond to the category's survey. At Tracksuit, this means the set of questions that the customer of the category requested. For simplicity, we have omitted the specific questions for this task and provided an estimated `category_length_seconds` variable to represent the expected time taken for the mean respondent to complete the category survey.

For this task, assume that we guarantee customers at least n=2,400 qualified respondents who complete the category survey each year. Since we do monthly surveys, this translates to roughly n=200 per month. Of course, the number of respondents required to achieve the desired qualified respondents per month varies by incidence rate. For a category with an incidence rate of 10%, we'd need to survey roughly `200/10%=2000` respondents in order to receive 200 qualified respondents to complete the category survey. If a survey mechanism didn't require all qualifying survey respondents to complete the category survey, the required respondent number would increase even more.

We provide an example of a set of categories and their associated information in the spreadsheet `fake_category_data.csv`. You may use this data to assist in validating your algorithm.

Here is the associated data dictionary:

**_Data dictionary_**
- category_id: the unique id for a category
- category_name: the name for the category
- incidence_rate: the estimated proportion of respondents from a population who would select a `qualifying` response to the category qualifier. For simplicity, we have omitted the category qualifiers and answer options themselves.
- category_length_seconds: the estimated time taken for the mean respondent to complete the category survey.

We hope you enjoy this take home task. If you have any questions - please reach out! We look forward to reviewing your submission.

