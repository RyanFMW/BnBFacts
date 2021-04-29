library(tidyverse)
library(topicmodels) # for model
library(tidytext) # for tokenizing 
library(data.table)
library(ldatuning) # for number of topics
library(udpipe) # for recode ngrams
library(ggrepel) # wrapped labels on scatterplot 

# get data (downloaded from Inside Airbnb and saved locally)

temp = list.files("C:\\Users\\Ryan\\Desktop\\US_2020_reviews", full.names = TRUE)
myfiles = lapply(temp, read_csv)

# filter reviews from only the last year as they will be used with ratings which are calculated using previous 12m

reviews.df <- do.call(rbind.data.frame,myfiles)  
reviews.df <- reviews.df %>% filter(date > "2019-02-27")

rm(myfiles)

# Identify and remove non-English reviews 

reviews.df$language_detect <- str_trunc(reviews.df$comments, 100, side = c("right"), ellipsis = "")

split <- split(reviews.df, f = (as.numeric(rownames(reviews.df))-1) %/% 10000)

language.output <- lapply(split, function(x) {                         
  x %>%
    mutate(language=detect_language(language_detect)) %>%
    select(id, language)}) 

language.df <- do.call(rbind.data.frame,language.output)  
rm(language.output)

reviews.df <- reviews.df %>% 
  inner_join(language.df) %>%
  filter(language == "en" | is.na(language))

rm(language.df)

# 3431614 reviews down to 3236554 when stripping out non-English

# run bing and afinn sentiments on remaining reviews. Chunk the data because it's big

split <- split(reviews.df, f= (as.numeric(rownames(reviews.df))-1) %/% 50000)

sentiment.output <- lapply(split, function(x) {                         
reviews.tokenized <- x %>%
  unnest_tokens(word, comments)
reviews.tokenized %>%
  inner_join(get_sentiments("bing")) %>% 
  select(id, sentiment) %>%
  group_by(id, sentiment) %>%
  summarise(count=n()) %>%
  spread(sentiment, count, fill=0) %>%
  mutate(bing_sentiment=positive-negative) %>%
  select(id, bing_sentiment)})

sentiment.df <- do.call(rbind.data.frame,sentiment.output)  

rm(split)
rm(sentiment.output)

reviews.df <- reviews.df %>%
  merge(sentiment.df, by="id", all=TRUE)

rm(sentiment.df)

split <- split(reviews.df, f= (as.numeric(rownames(reviews.df))-1) %/% 50000)

sentiment.output <- lapply(split, function(x) {                         
  reviews.tokenized <- x %>%
    unnest_tokens(word, comments)
  reviews.tokenized %>%
    inner_join(get_sentiments("afinn")) %>% 
    select(id, value) %>%
    group_by(id) %>%
    summarise(afinn_sentiment=sum(value))})

sentiment.df <- do.call(rbind.data.frame,sentiment.output)  

rm(split)
rm(sentiment.output)

reviews.df <- reviews.df %>%
  merge(sentiment.df, by="id", all=TRUE)

rm(sentiment.df)

# save output of all of the above because it was time intensive and to be used later

write_csv(reviews.df, "sentiment_reviews.csv")

# reload scored review data along with listing data from cluster analysis

listings.df <- read_csv("sentiment_listings.csv", col_names=TRUE)
reviews.df <- read_csv("sentiment_reviews.csv",col_names=TRUE)

# plot distribution of Afinn sentiment

sample_n(reviews.df,10000) %>%
  mutate(afinn_sentiment=replace_na(afinn_sentiment,0)) %>%
  ggplot(aes(x=afinn_sentiment)) +
  geom_density(fill="#5E17EB", size=0.8, alpha=0.5) +
  geom_vline(xintercept = 0, linetype="dashed") +
  xlim(-10,30) +
  labs(title = "Distribution of Reviews by Sentiment Score",
       subtitle = "Reviews on Airbnb overwhelmingly skew positive",
       x = "Sentiment Score",
       y = "Density") +
  theme(panel.background = element_rect(fill="white", linetype = "solid", color="black"),
        panel.grid.major = element_line( linetype = "blank"),
        panel.grid.minor = element_line( linetype = "blank"),
        plot.title = element_text(hjust = 0.5, size=18),
        plot.subtitle = element_text(hjust = 0.5, size=14),
        axis.text = element_text(size=10),
        axis.title = element_text(size=14))

# Afinn sentiment versus overall rating chart

reviews.df %>%
  mutate(afinn_sentiment=replace_na(afinn_sentiment,0)) %>%
  group_by(listing_id) %>%
  summarise(avg_sentiment=mean(afinn_sentiment)) %>%
  inner_join(listings.df, by = c("listing_id" = "id")) %>%
  mutate(review_scores_rating=review_scores_rating/20) %>%
  filter(review_scores_rating>=3.5) %>%
  group_by(review_scores_rating) %>%
  summarise(avg_sentiment=mean(avg_sentiment), count=n()) %>%
  ggplot(aes(x=review_scores_rating, y=avg_sentiment)) +
  geom_line(size = 1.5, colour="#5E17EB") +
  geom_point(size = 4, fill="#5E17EB", colour="#5E17EB") + 
  labs(title = "Overall Rating vs Average Sentiment",
       subtitle = "Higher sentiment scores translate to higher overall ratings",
       x = "Overall Rating",
       y = "Average Sentiment") +
  theme(panel.background = element_rect(fill="white", linetype = "solid", color="black"),
        panel.grid.major = element_line(linetype = "solid", color="grey80"),
        panel.grid.minor = element_line(linetype = "solid", color="grey80"),
        plot.title = element_text(hjust = 0.5, size=18),
        plot.subtitle = element_text(hjust = 0.5, size=14),
        axis.text = element_text(size=10),
        axis.title = element_text(size=14),
        legend.title = element_blank(),
        legend.text = element_text(size=10),
        legend.position = "top")

# Compare ratings distributions of listings with and without negative reviews. 

has_negative <- reviews.df %>%
  mutate(afinn_sentiment=replace_na(afinn_sentiment,0)) %>%
  mutate(sentiment_category = if_else(afinn_sentiment< -3, "negative", "non_negative")) %>%
  group_by(listing_id, sentiment_category) %>%
  summarise(count=n()) %>%
  spread(sentiment_category,count) %>%
  mutate(negative=replace_na(negative,0)) %>%
  mutate(non_negative=replace_na(non_negative,0)) %>%
  mutate(has_negative_review=if_else(negative>0,"Has Negative Review(s)", "No Negative Reviews")) %>%
  inner_join(listings.df, by = c("listing_id" = "id")) %>%
  select(listing_id, has_negative_review, review_scores_rating) %>%
  mutate(review_scores_rating=review_scores_rating/20) 

has_negative <- as.data.frame(has_negative)

rbind.data.frame(
  filter(has_negative, has_negative_review == "Has Negative Review(s)"), 
  sample_n(filter(has_negative, has_negative_review == "No Negative Reviews"),10000)) %>%
  ggplot(aes(x=review_scores_rating, fill=has_negative_review)) +
  geom_density(alpha=0.4) +
  xlim(3.5,5) +
  labs(title = "Distribution of Overall Rating",
       subtitle = "Listings with no negative reviews skew much higher in overall rating",
       x = "Overall Rating",
       y = "Density") +
  theme(panel.background = element_rect(fill="white", linetype = "solid", color="black"),
        panel.grid.major = element_line( linetype = "blank"),
        panel.grid.minor = element_line( linetype = "blank"),
        plot.title = element_text(hjust = 0.5, size=18),
        plot.subtitle = element_text(hjust = 0.5, size=14),
        axis.text = element_text(size=10),
        axis.title = element_text(size=14),
        legend.title = element_blank(),
        legend.text = element_text(size=10),
        legend.position = "top")


# quantify negative reviews as afinn score < -3, filter, unnest and recode tokens

neg.reviews.tokens <- reviews.df %>%
  filter(afinn_sentiment < -3) %>% 
  inner_join(listings.df, by = c("listing_id" = "id")) %>%
  filter(state != "HI", room_type == "Entire home/apt") %>%
  unnest_tokens(word, comments) %>%
  filter(str_detect(word,"[:digit:]")==FALSE) %>%
  mutate(word=str_replace(word, "beds", "bed"))

compound <- c("check in", "check out", "air bnb")
ngram <- c(2,2,2)

neg.reviews.tokens$word <- txt_recode_ngram(neg.reviews.tokens$word, compound, ngram, sep = " ")

neg.reviews.tokens <- neg.reviews.tokens %>% filter(!is.na(word)) %>% anti_join(stop_words)

# create dtm and filter unuseful terms 

neg.reviews.dtm <- neg.reviews.tokens %>%
  count(id, word, sort = TRUE) %>%
  filter(!word %in% c(
    "stay",
    "space",
    "host",
    "cabin",
    "house",
    "airbnb",
    "stay",
    "condo",
    "apartment",
    "issue",
    "home",
    "property",
    "hosts",
    "night",
    "issues",
    "experience",
    "guest",
    "guests",
    "air bnb",
    "day",
    "la",
    "de")) %>%
  cast_dtm(id, word, n)

# Identify number of topics to look for 

result <- FindTopicsNumber(neg.reviews.dtm ,
                           topics = seq(from = 5, to = 100, by = 5),
                           metrics = c("Griffiths2004", "CaoJuan2009", "Arun2010", "Deveaud2014"),
                           mc.cores = 4)

# Plot to identify number of topics

FindTopicsNumber_plot(result)


# Create model 

neg.reviews.lda <- LDA(neg.reviews.dtm, k = 40, control = list(seed = 1234))

neg.reviews.topic <- tidy(neg.reviews.lda, matrix = "beta")

top_terms <- neg.reviews.topic %>%
  group_by(topic) %>%
  top_n(10, beta) %>%
  ungroup() %>%
  arrange(topic, -beta)

# Generate topic names 

second_term_value <- top_terms %>%
  group_by(topic) %>% 
  slice_max(beta, n=2) %>%
  mutate(rank=rank(-beta)) %>%
  select(topic, beta, rank) %>%
  spread(rank, beta) %>%
  mutate(second_term_value=`2`/`1`)

topic_name <-  top_terms %>%
  group_by(topic) %>% 
  slice_max(beta, n=2) %>%
  mutate(rank=rank(-beta)) %>%
  select(-beta) %>%
  spread(rank, term) %>%
  inner_join(second_term_value, by =c("topic")) %>%
  mutate(topic_name = if_else(second_term_value<0.5,`1.x`,paste(`1.x`,`2.x`,sep =", "))) %>%
  mutate(topic_name = paste(topic, topic_name, sep = ") ")) %>%
  select(topic, topic_name)

# Chart once with auto-generated names 

top_terms %>%
  inner_join(topic_name, by =c("topic")) %>%
  mutate(term = reorder_within(term, beta, topic_name)) %>%
  ggplot(aes(beta, term, fill = factor(topic_name))) +
  geom_col(show.legend = FALSE) +
  facet_wrap(~ fct_reorder(topic_name, topic), scales = "free") +
  scale_y_reordered() +
  theme(axis.text.x=element_blank())

neg.reviews.topic2 %>%
  inner_join(topic_name, by = c("topic")) %>%
  filter(gamma > 0.1) %>%
  count(topic, topic_name) %>%
  mutate(topic_name=fct_reorder(topic_name,n)) %>%
  ggplot(aes(x=topic_name, y=n)) +
  geom_col() +
  coord_flip() 


# Chart again, filtering out select topics and renaming remaining topics

new_topic_names <- data.frame(topic= c(4,6,8,10,11,12,15,17,19,20,21,26,30,31,32,35,40), 
                              topic_name = c("Broken items", 
                                "Unsafe neighborhood",
                                "Inaccurate amenities",
                                "Dirty",
                                "Bathroom",
                                "Uncomfortable",
                                "Loud noises",
                                "Front door",
                                "Check in",
                                "Host interaction",
                                "Bed",
                                "Smell",
                                "Late to respond",
                                "Room temperature",
                                "Shower",
                                "Kitchen",
                                "Parking"))

top_terms %>%
  inner_join(new_topic_names, by =c("topic")) %>%
  mutate(term = reorder_within(term, beta, topic_name)) %>%
  ggplot(aes(beta, term, fill=reorder(topic_name,topic))) +
  geom_col(show.legend = FALSE) +
  facet_wrap(~ fct_reorder(topic_name, topic), scales = "free", ncol=3) +
  scale_y_reordered() +
  theme(axis.text.x=element_blank(),
        axis.title.x = element_blank(),
        axis.title.y = element_blank()) 

# chart by frequency of complaint; gamma > 0.1. most negative reviews contain multiple complaints thus 0.1

neg.reviews.topic2 %>%
  inner_join(new_topic_names, by = c("topic")) %>%
  filter(gamma > 0.1) %>%
  count(topic, topic_name) %>%
  mutate(topic_name=fct_reorder(topic_name,n)) %>%
  ggplot(aes(x=topic_name, y=n, fill = reorder(topic_name,topic))) +
  geom_col() +
  coord_flip() +
  labs(title = "Frequency of Topics in Negative Reviews",
       subtitle = "Top topics represent most common reason for negative reviews",
       x = element_blank(),
       y = "Frequency") +
  theme(plot.title = element_text(hjust = 0.5, size=18),
        plot.subtitle = element_text(hjust = 0.5, size=14),
        axis.title = element_text(size=14),
        legend.position = "none")

# Assign topics to reviews 

neg.reviews.topic2 <- tidy(neg.reviews.lda, matrix = "gamma")

# chart by severity (average negative sentiment)

severity <- neg.reviews.topic2 %>%
  inner_join(new_topic_names, by = c("topic")) %>%
  mutate(document=as.numeric(document)) %>%
  inner_join(reviews.df, by = c("document" = "id")) %>%
  filter(gamma > 0.1) %>%
  group_by(topic, topic_name) %>%
  summarise(avg_sentiment=mean(afinn_sentiment)*-1) 

severity %>%
  ggplot(aes(x=fct_reorder(topic_name,avg_sentiment), y=avg_sentiment, fill=reorder(topic_name,topic))) +
  geom_col() +
  coord_flip(ylim = c(5,11), xlim =c(-1,17)) +
  geom_rect(ymin=4.5, ymax=11.5, xmin=-1.6, xmax=0, fill = "white") +
  annotate("text", x = -0.4, y = 10, label = "High", size =3.5) +
  annotate("text", x = -0.4, y = 8, label = "Mid", size =3.5) +
  annotate("text", x = -0.4, y = 6, label = "Low", size =3.5) +
  annotate("text", x = -1.2, y = 8, label = "Severity", size =4.5) +
  labs(title = "Severity of Topics in Negative Reviews",
       subtitle = "Top topics represent issues that make guests the angriest",
       x = element_blank(),
       y = "Severity") +
  theme(plot.title = element_text(hjust = 0.5, size=18),
        plot.subtitle = element_text(hjust = 0.5, size=14),
        axis.title = element_text(size=14),
        axis.text.x = element_blank(),
        axis.ticks.x = element_blank(),
        axis.title.y = element_blank(),
        axis.title.x = element_blank(),
        legend.position = "none")

# chart by impact to overall rating

impact <- neg.reviews.topic2 %>%
  inner_join(new_topic_names, by = c("topic")) %>%
  mutate(document=as.numeric(document)) %>%
  inner_join(reviews.df, by = c("document" = "id")) %>%
  inner_join(listings.df, by = c("listing_id" = "id")) %>%
  filter(gamma > 0.1) %>%
  group_by(topic, topic_name) %>%
  summarise(avg_overall_rating=mean(review_scores_rating/20), count=n())

impact %>%
  ggplot(aes(x=fct_reorder(topic_name,avg_overall_rating, .desc = TRUE), y=avg_overall_rating, fill=reorder(topic_name,topic))) +
  geom_col() +
  coord_flip(ylim = c(4.3,4.55)) +
  labs(title = "Negative Review Topics vs Overall Rating",
       subtitle = "Topics at the top appear in lower rated listings",
       x = element_blank(),
       y = "Overall Rating") +
  theme(plot.title = element_text(hjust = 0.5, size=18),
        plot.subtitle = element_text(hjust = 0.5, size=14),
        axis.title = element_text(size=14),
        legend.position = "none")


# severity x impact chart

impact %>% 
  inner_join(severity, by = c("topic")) %>%
  ggplot(aes(x=avg_overall_rating, y=avg_sentiment)) +
  geom_point() +
  geom_smooth(method="lm", linetype=3, col="black", se=FALSE, lwd = 1) +
  geom_label_repel(aes(label=topic_name.x, fill=reorder(topic_name.x, topic))) +
  annotate("text", x = 4.37, y = 10, label = "High", angle =90) +
  annotate("text", x = 4.37, y = 8.5, label = "Mid", angle =90) +
  annotate("text", x = 4.37, y = 7, label = "Low", angle =90) +
  annotate("text", x = 4.36, y = 8.5, label = "Severity", angle =90, size = 5) +
  coord_cartesian(xlim = c(4.38, 4.53), clip = "off") +
  labs(title = "Overall Rating vs Severity of Complaint",
       subtitle = "Negative correlation shows more severe complaints lead to lower ratings",
       x = "Overall Rating",
       y = "Severity") +
theme(plot.margin = margin(0.5, 0.5, 0.5, 2, "cm"),
      axis.ticks.y = element_blank(),
      axis.text.y = element_blank(),
      plot.title = element_text(hjust = 0.5, size=18),
      plot.subtitle = element_text(hjust = 0.5, size=14),
      axis.title.x = element_text(size=14),
      axis.title.y = element_blank(),
      legend.position = "none")
  




