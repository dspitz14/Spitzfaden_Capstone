#Proposal 1:Predicting Restaurant Health Inspection Scores from Yelp Reviews and Ratings

##Desciption:
For small, local restaurants, maintaining a high health inspection rating is critical to reputation. However, inspections are random and owners are unable to know when inspectors will visit. Therefore, it's important for owners to have a tool to estimate how their restaurant will perform at the next inspection. This project utilizes Yelp reviews and ratings to predict how a restaurant will perform.

##Research Question(s) or Objective(s):
-How will a restaurant perform on its next health inspection?

##Final Presentation:
-Presentation and Website

##Data Sources:
-Health Inspection Scores: data.austintexas.gov
-Reviews and Ratings: Yelp API (Potentially other digital mentions of restaurants)
-Seems like there are other city data on Kaggle

##Challenges:
-It seems like health inspections are way more frequent than I thought. I'm not sure Yelp reviews are frequent enough to predict each and every score. I'd likely have to bin data in weird ways.

##Related Areas to Explore
-Improved restaurant recommender based on power users or users that have similar ratings to you
-Conversely, health inspectors can use data to target restaurants that could be at risk



#Proposal 2: Bike Share Demand Estimation for Mid-Size Cities

##Desrciption:
Every major American city has a bike share program. Small towns are unlikely to have bike share programs. This project helps mid-size cities determine if a program is viable by looking at bike share membership/ride data from existing programs and MSA data on the cities where those programs exist.

##Research Question(s) or Objective(s):
-Is a bike share program viable in certain cities?
-What price should a new bike share program set?
-How do members and non-members use bike shares differently?
-How do different cities use bike share programs?
-Will a city provide its own bike share program or will a company like B-cycle run it?


##Final Presentation:
-Presentation and Website

##Data Sources:
-For Austin- data.austintexas.gov
-For other cities- I have seen similar data for various cities, but would have to dig
-MSA data- census.gov or bls.gov

##Challenges:
-Getting a complete dataset for multiple cities seems incredibly challenging. The various city data will be spread out and likely not be in the same format. Price data will likely be manual data collection on each programs website.
-Estimating demand is generally an extremely hard thing to measure. I'd need to think about substitute goods and figure out what those prices are in every city I'm interested in. There are likely other features I'm not even thinking about as well. Weather trends? etc.

##Related Areas to Explore



#Proposal 3: Measuring the Impact of Russian Accounts on the 2016 Election

##Description: The further removed form the 2016 election we are, the more clear it is that Russia interfered in some way. However, it has been hard to measure the effect Russian interference had on the election. This project aims to quantify interference by looking at social media posts related to the election and the amount of engagement (and type of engagement) with comments made before and after known Russian accounts interacted with the posts.

##Research Question(s) or Objective(s):
-How effective were Russian accounts in creating disparity among American Twitter and Facebook users?
-Were certain demographics of Americans more susceptible to Russian interference?
-How did conversation on threads change from before a Russian account commented to after?
-When did Russian accounts begin interfering?
-Did Russian accounts influence key election topics?
-Did people who interacted with Russian accounts later echo the accounts in posts?


##Final Presentation:
-Presentation and Website

##Data Sources:
-Twitter API
-Facebook API
-Potentially a tool like TrendKite to see how topics were covered in mainstream media

##Challenges:
-Known Russian accounts have likely been suspended or deleted, making it difficult to fetch their posts.
-Social media data is notoriously messy; Facebook replies to posts are incredibly messy to figure out who replied to whom, but I have done it before
-Correlating more social conversation to poll results is a bit of a stretch

##Related Areas to Explore
-Looking at effect on elections down the ballot
-NLP on various camps in the conversations
-How do Facebook reactions differ between different groups?


#Other Ideas that I've thought less about
-Predict if an NCAA basketball player will transition well to the NBA
-Predict if a bill will pass through congress based on the text of the bill
-Predict Oscar Best Picture Winner based on IMDB ratings, Rotten Tomatoes ratings, Academy member Twitter mentions, box office revenue, cast member accolades, other prestigious award noms and wins
-Predict invasive species risk in WI/MN lakes using satellite images
-Predict Trump cabinet terminations from Trump Twitter activity
-Predict LEGO brick price on bricklink.com using rebrickable.com data (didn't realize that stuff existed)
-Create an improved CPI looking at prices from Amazon
-Segment customer base based on Twitter behavior (followership, chi-squared on content)
