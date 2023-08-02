# ![](https://ga-dash.s3.amazonaws.com/production/assets/logo-9f88ae6c9c3871690e33280fcf557f33.png) Capstone Project

### Problem Statement

As a data analyst at the Singapore Tourism Board, my primary responsibility is to collect and label data, conduct an extensive analysis and create a sophisticated machine learning model aimed at predicting the Net Promoter Score (NPS) of diverse tourist attractions in Singapore. This predictive model will utilize TikTok posts and comments pertaining to these attractions as its basis.

The overarching goal is to equip the Singapore Tourism Board with valuable insights and a comprehensive understanding of trends and sentiments expressed on this nascent social media platform. By leveraging these insights, the board can make well-informed, data-driven decisions, develop targeted strategies, and assist relevant stakeholders in the tourism industry (e.g. Tourist Attractions Management, Travel Agencies and Tour Operators) to improve their services and offerings. Ultimately, these efforts will enhance the overall appeal of Singapore as a premier global tourist destination.

### Objectives

1. Collect and label training data to train the predictive model.

2. Analyze and interpret sentiment trends concerning different tourist attractions based on TikTok posts and comments.

3. Develop a robust machine learning model that accurately predicts the NPS of Singapore's tourist attractions.

---

### Data Dictionary

`video_info_audio_caption_cleaned_df`

|Feature|Type|Description|
|:---|:---:|:---|
|<b>id</b>|*object*|Id of the TikTok post|
|<b>url</b>|*object*|URL address of the TikTok post|
|<b>account_name</b>|*object*|Account name of the TikTok post uploader|
|<b>following_count</b>| *int64*|Following count of the TikTok post uploader|
|<b>follower_count</b>|*object*|Follower count of the TikTok post uploader|
|<b>total_like_count</b>|*object*|Total like count of the TikTok post uploader|
|<b>date</b>|*object*|Date on which the TikTok post was uploaded|
|<b>href</b>|*object*|The href needed to access the link to the TikTok post uploader's account page|
|<b>handle</b>|*object*|TikTok post uploader's account handle|
|<b>description</b>|*object*|Description of the TikTok post|
|<b>hashtag</b>|*object*|Hashtags of the TikTok post|
|<b>like_count</b>|*object*|Like count of the TikTok post|
|<b>bookmark_count</b>|*object*|Bookmark count of the TikTok post|
|<b>share_count</b>|*object*|Share count of the TikTok post|
|<b>comment_count</b>|*object*|Comment count of the TikTok post|
|<b>final_text</b>|*object*|Text from speech-to-text, unless empty, and if empty, it will be text from caption-to-text |

<br>

`comments_df`

|Feature|Type|Description|
|:---|:---:|:---|
|<b>id</b>|*object*|Id of the post with indication of being a comment instead of post|
|<b>url</b>|*object*|URL address of the TikTok post|
|<b>handle</b>|*object*|TikTok post uploader's account handle|
|<b>comment_count</b>| *object*|Comment count of the TikTok post|
|<b>comment</b>|*object*|Comment text|

<br>

`train_df`

|Feature|Type|Description|
|:---|:---:|:---|
|<b>post_comment</b>|*object*|Whether this sentence is from post or comment|
|<b>id</b>|*object*|Id of the TikTok post with indication of whether the sentence is from a post or comment|
|<b>sentence</b>|*object*|Sentence text|
|<b>entity</b>| *object*|Entities found in the sentence|
|<b>pos_sentiment_words</b>|*object*|Words found in the sentence with positive sentiment|
|<b>neg_sentiment_words</b>|*object*|Words found in the sentence with negative sentiment|
|<b>textblob</b>|*float64*|Sentiment score ranging from -1 to 1|
|<b>sentiment</b>|*float64*|Adjusted sentiment score ranging from 0 to 10|

<br>

`sg_entities_patterns_df`
Note: this dataset was manually prepared and not scrapped.

|Feature|Type|Description|
|:---|:---:|:---|
|<b>label</b>|*object*|Label of the tourist attraction (entity)|
|<b>pattern</b>|*object*|Pattern to recognize as tourist attraction (entity)|
|<b>sublocation</b>|*object*|Actual name of the tourist attraction (entity)|
|<b>interest_1</b>| *object*|Main category in which the tourist attraction falls under|
|<b>interest_2</b>|*object*|Secondary category in which the tourist attraction falls under|
|<b>indoor_outdoor</b>|*object*|Whether the tourist attraction is indoors or outdoors|

<br>

---

### Business Recommendations

Contextualising the insights for the various stakeholders in the Singapore tourism industry, here are a few business recommendations to consider:

1. Costliness and the overall impression of poor value for money is a major concern for tourist attractions and shopping in Singapore. Stakeholders should consider if the high price points are a nett positive or negative revenue driver for their portfolio, both in the short and long term.

2. In terms of social media marketing, efficacy increases greatly with eye-catching one-of-a-kind features (e.g. SIA suites) or easy-to-mention awards. These help to drive virality, reach and engagement.

3. Singapore excels in areas where tourists have comparative experiences (e.g. Singapore Airport, Singapore Airlines), where the Singapore experience is superior (but not necessarily unique or defining) to others. These are not a sufficient draw to increase tourism numbers or revenue. To further improve on Singapore’s tourism prospects, other areas (e.g. quality of food, value of shopping, cultural experience) need improvement.

The machine learning model will prove invaluable in providing assistance and valuable insights to different stakeholders in the tourism industry:

- Tourist Attractions Management: Track the progress of their NPS. By leveraging on NPS, they would be able to understand and optimize their customer service and deliverables to increase footfall and revenue. They can also use this tool to monitor competitors’ performance.

- Singapore Tourism Board: Track NPS of different attractions to assess overall visitor satisfaction and target improvements in the tourism sector. NPS would also serve as an indicator of the attractiveness and potential success of a tourist attraction, influencing their subsequent investment decisions when building or supporting new tourist destinations.

- Travel Agencies and Tour Operators: Track NPS of various attractions to design better travel packages and recommend attractions that align with the current trends

- In addition, stakeholders can track the efficacy of specific campaigns or initiatives by analyzing the pre- and post-frequency of mentions of the particular entity. This enables them to discern the impact of their initiatives and make data-driven decisions to optimize promotional strategies and engagement with tourists.

---

### Conclusion

Upon revisiting the problem statement, we are reminded of our three primary objectives:

1) <b>Objective 1</b>
- Our first objective was to collect and label training data to train the predictive model.
- The successful collection and labeling of training data have facilitated the development of a robust machine learning model capable of accurately predicting the NPS for various tourist attractions in Singapore.

2) <b>Objective 2</b>
- Our second objective was to analyze and interpret sentiment trends concerning different tourist attractions based on TikTok posts and comments. 
- We found 3 main points:
    - Costliness and poor value for money concern tourist attractions and shopping in Singapore. 
    - Eye-catching features and easy-to-mention awards enhance social media marketing. 
    - Singapore excels in areas with comparative advantages, but improving food, shopping, and cultural experiences is essential for further enhancing tourism prospects.


3) <b>Objective 3</b>
- Our third objective was to develop a robust machine learning model that accurately predicts the NPS of Singapore's tourist attractions.
- With a test MSE score of 0.0012, test RMSE score of 0.034 and test R^2 score of 0.91, the final model demonstrates a high level of accuracy in predicting the NPS for diverse tourist attractions in Singapore. This proficient model can serve as a valuable tool for various stakeholders in presenting actionable insights to the management of their respective companies.

---

### Next Steps

1) <b>Broaden Dataset</b>  
- this would make the model more robust and representative (e.g. more posts, other socials like Instagram, FaceBook posts, etc.)
2) <b>Introduce Source Identification</b> 
- this allows for more granular deep-dives and stronger root cause analysis (e.g. nationality/gender/age of creator, social platform drawn from, post or comment, etc.)
3) <b>Explore NLP Tools</b>
- research other sentiment analysis python libraries to refine labelling of scores, Dependency Parsing to extract descriptive words associated to a specific entity
