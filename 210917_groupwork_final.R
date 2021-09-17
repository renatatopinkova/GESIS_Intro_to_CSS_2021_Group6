# Topics and sentiments on r/GradSchool reddit in the last year

# 1 Load libraries -------------------------------------------------------------

# install.packages("reshape2")

# install these packages if needed before

library(ggplot2)
library(RedditExtractoR)
library(syuzhet)
library(tidyverse)
library(tidytext)
library(topicmodels)
library(quanteda)

# 2 Scrape data ----------------------------------------------------------------

# get urls of threads of r/GradSchool

urls <- find_thread_urls(subreddit = "GradSchool", 
                         sort_by = "top", 
                         period = "year")

print(paste("Number of rows:", nrow(urls))) # 995

# get content of threads and save in data frame

# because there is a "limit" for each call of get_thread_content(), a loop is 
# used to get the content of each thread separately and the resulting data 
# frames are combined using rbind()

# first row

content <- get_thread_content(urls$url[1])
df <- data.frame(content$threads)

# 2nd to nth row

for (row in 2:nrow(urls)) {
  tryCatch({
    print(row)
    content <- get_thread_content(urls$url[row])
    df <- rbind(df, data.frame(content$threads))
  }, error=function(e){cat("ERROR :",conditionMessage(e), "\n")})
}

# data frame includes fewer rows than threads ulrs because for some thread urls
# there was an error message with the get_thread_content() which was skipped
# using tryCatch()

print(paste("Number of rows:", nrow(df))) # 995 rows, no rows removed

# save data frame

save(df, file = "df.Rda")

# 3 Pre-processing ----------------------------------------------------------

# load and explore data frame

load("df.Rda")

glimpse(df)

# get baseform data for lemmatization

lemma_data <- read.csv("baseform_en.csv", encoding = "UTF-8")

# assign ids to rows 

df$id <- seq.int(nrow(df))

# remove texts that have less than 25 characters 
# (i.e., empty, pictures only, removed/deleted)

df <- df %>%
  filter(str_length(text) > 25)

print(paste("Number of rows:", nrow(df))) # 925 rows, 70 rows removed

### pre-process texts (thread posts)

# corpus and dfm for texts

dfm_texts <- corpus(df$text, docnames = df$id) %>%
  # remove punctuation, numbers, symbols, and urls
  tokens(.,remove_punct=TRUE, remove_numbers=TRUE, remove_symbols = TRUE,
         remove_url = TRUE) %>%
  # convert to lowercase
  tokens_tolower() %>% 
  # lemmatize
  tokens_replace(lemma_data$inflected_form, lemma_data$lemma, 
                 valuetype = "fixed") %>%
  # remove stopwords and words with less than 3 chars
  tokens_select(., pattern = stopwords("en"), 
                min_nchar = 3, selection = "remove") %>%
  # convert to document-feature-matrix
  dfm() %>%
  # remove texts that are empty after pre-processing
  dfm_subset(., ntoken(.) > 0)

dfm_texts # 919 documents (thread posts), 6 rows excluded

### pre-process titles (thread titles)

# corpus and dfm for titles

dfm_titles <- corpus(df$title, docnames = df$id) %>%
  # remove punctuation, numbers, symbols, and urls
  tokens(.,remove_punct=TRUE, remove_numbers=TRUE, remove_symbols = TRUE,
         remove_url = TRUE) %>%
  # convert to lowercase
  tokens_tolower() %>% 
  # lemmatize
  tokens_replace(lemma_data$inflected_form, lemma_data$lemma, 
                 valuetype = "fixed") %>%
  # remove stopwords and words with less than 3 chars
  tokens_select(., pattern = stopwords("en"), 
                min_nchar = 3, selection = "remove") %>%
  # convert to document-feature-matrix
  dfm() %>%
  # remove titles that are empty after pre-processing
  dfm_subset(., ntoken(.) > 0)

dfm_titles # 917 documents (thread titles), 8 rows excluded

# 4 Topic models ---------------------------------------------------------------

# set n of topics
K <- 10

### LDA for texts (thread posts)

# compute the model

lda_texts <- LDA(dfm_texts, k = K, method = "Gibbs", 
                 control = list(verbose=25L, seed = 123, burnin = 100,
                                iter = 500))

# show main terms in topics

terms_texts <- get_terms(lda_texts, 10)
terms_texts

# show main topic for texts

topics_texts <- get_topics(lda_texts, 1)
head(topics_texts)

# get examples

paste(terms_texts[,1], collapse=", ") # terms for topic 1
sample(df$text[topics_texts==1], 3) # example texts for topic 1

### LDA for titles (thread titles)

# compute the model

lda_titles <- LDA(dfm_titles, k = K, method = "Gibbs", 
                  control = list(verbose=25L, seed = 123, burnin = 100,
                                 iter = 500))

# show main terms in topics

terms_titles <- get_terms(lda_titles, 10)
terms_titles

# show main topic for titles

topics_titles <- get_topics(lda_titles, 1)
head(topics_titles)

# get examples

paste(terms_titles[,1], collapse=", ") # terms for topic 1
sample(df$title[topics_titles==1], 3) # example texts for topic 1

# visualize results of topic models

### graph for texts (thread posts)

# get betas - probabilities for words to be associated with topic

beta_texts <- tidy(lda_texts, matrix = "beta")

# wrangling

ten_top_terms_texts <- beta_texts %>%
  group_by(topic) %>%
  slice_max(beta, n = 10) %>% 
  ungroup() %>%
  arrange(topic, -beta)

# ggplot

topic_names_texts <- c(
  `1` = "MA to PhD",
  `2` = "Time pressure",
  `3` = "Reading etc.",
  `4` = "Support ment. health",
  `5` = "Find prog. advisor",
  `6` = "Social life",
  `7` = "Progress work/skills",
  `8` = "Overworked",
  `9` = "Emotions",
  `10` = "Thesis work"
)

ten_top_terms_texts %>%
  mutate(term = reorder_within(term, beta, topic)) %>%
  ggplot(aes(beta, term, fill = factor(topic))) +
  geom_col(show.legend = FALSE) +
  facet_wrap(~ topic, scales = "free", 
             labeller = as_labeller(topic_names_texts)) +
  scale_y_reordered() +
  ggtitle("Ten topics based on thread posts - \nbetas of top ten terms")

ggsave("topics_terms_texts.png", width = 8, height = 5)

### graph for titles (thread titles)

beta_titles <- tidy(lda_titles, matrix = "beta")

# wrangling

ten_top_terms_titles <- beta_titles %>%
  group_by(topic) %>%
  slice_max(beta, n = 10) %>% 
  ungroup() %>%
  arrange(topic, -beta)

# ggplot

topic_names_titles <- c(
  `1` = "Phd-Advisor relations",
  `2` = "Find prog. advisor",
  `3` = "Time management",
  `4` = "Progress",
  `5` = "Covid struggles",
  `6` = "Finish work",
  `7` = "Advise ment. health",
  `8` = "Jobs, find job",
  `9` = "Gradschool",
  `10` = "(Find) social support"
)

ten_top_terms_titles %>%
  mutate(term = reorder_within(term, beta, topic)) %>%
  ggplot(aes(beta, term, fill = factor(topic))) +
  geom_col(show.legend = FALSE) +
  facet_wrap(~ topic, scales = "free", labeller = 
               as_labeller(topic_names_titles)) +
  scale_y_reordered() +
  ggtitle("Ten topics based on thread titles - \nbetas of top ten terms")

ggsave("topics_terms_titles.png", width = 8, height = 5)



# 5 Sentiment analysis for topics ---------------------------------------------------------

### sentiment analyses and graph for texts (thread posts)

# calculate sentiment for texts

df$sentiment_texts <- get_sentiment(df$text, method = "afinn")

topics_texts_df <- data.frame(topics_texts=unlist(topics_texts),
                              id=names(topics_texts))

# convert ids from char to int

topics_texts_df$id <- as.integer(topics_texts_df$id)

# merging to original data set

texts_merged <- inner_join(topics_texts_df, df, by = "id")

ggplot(texts_merged, aes(x=topics_texts, y=sentiment_texts)) + geom_point(mapping = aes(color=sentiment_texts)) +
  stat_summary(aes(y = sentiment_texts,group=1), fun=mean, colour="red", geom="line",group=1) + 
  labs(y="Sentiment Text/Topics", x="Topics") + 
  theme_bw() +
  ggtitle("Topic Models and Sentiment Scores") +
  scale_x_continuous(breaks = seq(0, 10, by = 1)) +
  scale_y_continuous(breaks = seq(-40, 40, by = 10)) +
  coord_flip()

ggsave("topics_sentiment_texts_distribution.png", width = 8, height = 5)  

# calculate mean sentiment for texts in each topic and plot it

sentiment_plot_texts <- texts_merged %>% 
  group_by(topics_texts) %>% 
  summarise(mean_sent = mean(sentiment_texts), 
            n=n()) %>%
  arrange(desc(mean_sent)) %>%
  mutate(order = as.numeric(rownames(.)))

ggplot(data = sentiment_plot_texts) +
  geom_bar(mapping = aes(x = reorder(topics_texts, order), y = mean_sent),
           stat = "identity") +
  labs(y = "mean sentiment") +
  scale_x_discrete("topics", labels = c("1" = "MA to PhD",
                                        "2" = "Time pressure",
                                        "3" = "Reading etc.",
                                        "4" = "Support ment. health",
                                        "5" = "Find prog. advisor",
                                        "6" = "Social life",
                                        "7" = "Progress work/skills",
                                        "8" = "Overworked",
                                        "9" = "Emotions",
                                        "10" = "Thesis work")) +
  coord_flip()

ggsave("topics_sentiment_texts.png", width = 8, height = 5)

### sentiment analyses and graph for titles (thread titles)

# calculate sentiment for titles

df$sentiment_titles <- get_sentiment(df$text, method = "afinn")

topics_titles_df <- data.frame(topics_titles=unlist(topics_titles),
                               id=names(topics_titles))

# convert ids from char to int

topics_titles_df$id <- as.integer(topics_titles_df$id)

# merging to original data set

titles_merged <- inner_join(topics_titles_df, df, by = "id")

ggplot(titles_merged, aes(x=topics_titles, y=sentiment_titles)) + geom_point(mapping = aes(color=sentiment_texts)) +
  stat_summary(aes(y = sentiment_titles,group=1), fun=mean, colour="red", geom="line",group=1) + 
  labs(y="Sentiment Titles/Topics", x="Topics") + 
  theme_bw() +
  ggtitle("Topic Models and Sentiment Scores") +
  scale_x_continuous(breaks = seq(0, 10, by = 1)) +
  scale_y_continuous(breaks = seq(-40, 40, by = 10)) +
  coord_flip()

ggsave("topics_sentiment_titles_distribution.png", width = 8, height = 5) 

# calculate mean sentiment for titles in each topic and plot it

sentiment_plot_titles <- titles_merged %>% 
  group_by(topics_titles) %>% 
  summarise(mean_sent = mean(sentiment_titles), 
            n=n()) %>%
  arrange(desc(mean_sent)) %>%
  mutate(order = as.numeric(rownames(.)))

# here we still need to relabel the numeric topics to meaningful names

ggplot(data = sentiment_plot_titles) +
  geom_bar(mapping = aes(x = reorder(topics_titles, order), y = mean_sent),
           stat = "identity") +
  labs(y = "mean sentiment") +
  scale_x_discrete("topics", labels = c("1" = "Phd-Advisor relations",
                                        "2" = "Find prog. advisor",
                                        "3" = "Time management",
                                        "4" = "Progress",
                                        "5" = "Covid struggles",
                                        "6" = "Finish work",
                                        "7" = "Advise ment. health",
                                        "8" = "Jobs, find job",
                                        "9" = "Gradschool",
                                        "10" = "(Find) social support")) +
  coord_flip()

ggsave("topics_sentiment_titles.png", width = 8, height = 5)


# Most frequent words -----------------------------------------------------
# tutorial here: https://tutorials.quanteda.io/statistical-analysis/frequency/

# require textstat function
require(quanteda.textstats)

dfm_texts %>% 
  # show top 20 words
  textstat_frequency(n = 20) %>% 
  ggplot(aes(x = reorder(feature, frequency), y = frequency)) +
  geom_point() +
  coord_flip() +
  labs(x = NULL, y = "Frequency") +
  theme_minimal()



# Calculate sentiment analysis for selected words -------------------------

# clean the dataframe 
df2 <- df %>%
  # filter out texts that have less than 25 characters
  filter(str_length(text) > 25) %>%
  # remove digits & punctuations
  mutate(text = str_replace_all(text, pattern="[0-9]+|[[:punct:]]|\\(.*\\)", "")) %>%
  # " ' " gets imported as \031 -> remove
  mutate(text = str_replace_all(text, pattern="\\031", "")) %>%
  # convert to lowercase
  mutate(text= tolower(text)) %>%
  # tokenize
  unnest_tokens(word, text) %>%
  # remove stopwords
  anti_join(stop_words) %>%
  # group again by id (=original post)
  group_by(id) %>% 
  # untokenize (= reverse antijoin)
  summarize(text = str_c(word, collapse = " ")) %>%
  ungroup() 
#TODO: lemmatization would be in order


# calculate sentiment analysis for posts that include selected word
# text[str_detect(text, "advisor") == returns rows of text that meet condition in []
df2 %>%
  summarise(advisor = round(mean(get_sentiment(text[str_detect(text, "advisor")], method="afinn")), 2),
            supervisor = round(mean(get_sentiment(text[str_detect(text, "supervisor")], method="afinn")), 2),
            thesis = round(mean(get_sentiment(text[str_detect(text, "thesis")], method="afinn")),2),
            phd = round(mean(get_sentiment(text[str_detect(text, "phd")], method="afinn")),2),
            work = round(mean(get_sentiment(text[str_detect(text, "work")], method="afinn")),2)) %>%
  # reshape from wide to long
  pivot_longer(c(advisor, supervisor, thesis, phd, work), names_to = "word", values_to = "value") %>%
  ggplot() +
  geom_bar(aes(x=word, y=value, fill = word), stat="identity") + 
  guides(fill=FALSE) + labs(x = "Word", y = "Sentiment value") + 
  theme_bw()


# overview of how many posts mention selected words
length(df2$text[str_detect(df2$text, "advisor")]) # 128 posts
length(df2$text[str_detect(df2$text, "supervisor")]) # 45 posts
length(df2$text[str_detect(df2$text, "thesis")]) # 107 posts
length(df2$text[str_detect(df2$text, "phd")]) # 261 posts
length(df2$text[str_detect(df2$text, "work")]) # 464
length(df2$text[str_detect(df2$text, "write")]) # 68 posts - clearly needs lemmatization