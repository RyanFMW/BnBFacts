library(tidyverse)
library(tidytext)
library(udpipe)
library(ggraph)
library(igraph)
library(ggforce)

# load data. data was scraped late January 2022
losangeles <- read_csv("losangeles.csv")
newyork <- read_csv("newyork.csv")
dallas <- read_csv("dallas.csv")
colorado_springs <- read_csv("colorado_springs.csv")
miami <- read_csv("miami.csv")

data <- rbind(losangeles, newyork, dallas, colorado_springs, miami)  
data$id = gsub("[^0-9]", "", data$url)

# parse description into seperate sections

data <- data %>% 
  separate(description, c("Description","Space"), sep=c("The space")) %>% 
  separate(Space, c("Space", "Guest_access"), sep=c("Guest access")) %>% 
  separate(Guest_access, c("Guest_access", "Other_things_to_note"), sep=c("Other things to note")) %>% 
  separate(Other_things_to_note, c("Other_things_to_note", "License"), sep=c("License number")) %>% 
  separate(Description, c("Description","Guest_access2"), sep=c("Guest access")) %>% 
  separate(Guest_access2, c("Guest_access2","Other_things_to_note2"), sep=c("Other things to note")) %>% 
  separate(Other_things_to_note2, c("Other_things_to_note2","License2"), sep=c("License number")) %>% 
  separate(Space, c("Space","Other_things_to_note3"), sep=c("Other things to note")) %>% 
  separate(Other_things_to_note3, c("Other_things_to_note3","License3"), sep=c("License number")) %>% 
  separate(Description, c("Description","Other_things_to_note4"), sep=c("Other things to note")) %>% 
  separate(Other_things_to_note4, c("Other_things_to_note4","License4"), sep=c("License number")) %>% 
  separate(Description, c("Description","License5"), sep=c("License number")) %>% 
  separate(Space, c("Space","License6"), sep=c("License number")) %>% 
  separate(Guest_access, c("Guest_access","License7"), sep=c("License number")) %>% 
  separate(Guest_access2, c("Guest_access2","License8"), sep=c("License number")) %>%
  unite("Guest_access", c(Guest_access, Guest_access2), remove = TRUE, na.rm=TRUE) %>% 
  unite("Other_things_to_note", c(Other_things_to_note, Other_things_to_note2, Other_things_to_note3, Other_things_to_note4), remove = TRUE, na.rm=TRUE) %>% 
  unite("License", c(License, License2, License3, License4, License5, License6, License7, License8), remove = TRUE, na.rm=TRUE) 

data <- mutate_all(data, list(~na_if(.,"")))
  
data <- data %>%
  mutate(has_Description = if_else(!is.na(Description), 1, 0),
         has_Space = if_else(!is.na(Space), 1, 0),
         has_Guest_access = if_else(!is.na(Guest_access), 1, 0),
         has_Other_things_to_note = if_else(!is.na(Other_things_to_note), 1, 0))

# graph showing completion rate

data %>% pivot_longer(cols = c(has_Description, has_Space, has_Guest_access, has_Other_things_to_note), names_to = "metric", values_to = "count") %>%
  group_by(metric) %>%
  summarise(percent = sum(count)/nrow(data)) %>%
  mutate(metric = fct_reorder(metric, percent)) %>% 
  ggplot(aes(x = percent, y = metric, fill = metric, label = scales::percent(percent))) +
  geom_col() + 
  geom_label(fill = "White", hjust = 0.75) +
  scale_y_discrete(labels = c("Other things to note", "Guest access", "The space", "Description")) +
  scale_x_continuous(labels = scales::percent) +
  labs(title = "Completion Rate of Airbnb Description Sections",
       subtitle = "Most hosts complete the description section",
       x = "Percent of listings to complete",
       y = "Section") +
  theme(panel.background = element_rect(fill="white", linetype = "solid", color="black"),
        panel.grid.major = element_line( linetype = "blank"),
        panel.grid.minor = element_line( linetype = "blank"),
        plot.title = element_text(hjust = 0.5, size=18),
        plot.subtitle = element_text(hjust = 0.5, size=14),
        axis.text = element_text(size=10),
        axis.title = element_text(size=14),
        legend.position = "None") +
  theme(plot.margin = unit(c(1,1,1,1), "cm"))

# Use same colours later 
description_color = "#C77CFF"
space_color = "#00BFC4"
guest_access_color = "#7CAE00"
other_things_color = "#F8766D"

# graph distribution of description length 

data %>%
  mutate(desc_length = nchar(Description)) %>%
  filter(desc_length <= 500) %>%
  ggplot(aes(x=desc_length)) +
  geom_histogram(fill = description_color) +
  labs(title = "Distribution of character length of Airbnb descriptions",
       subtitle = "Most hosts make use of the full space provided",
       x = "Character length of description section",
       y = "Count or listings") +
  theme(panel.background = element_rect(fill="white", linetype = "solid", color="black"),
        panel.grid.major = element_line( linetype = "blank"),
        panel.grid.minor = element_line( linetype = "blank"),
        plot.title = element_text(hjust = 0.5, size=18),
        plot.subtitle = element_text(hjust = 0.5, size=14),
        axis.text = element_text(size=10),
        axis.title = element_text(size=14),
        legend.position = "None") +
  theme(plot.margin = unit(c(1,1,1,1), "cm"))

# graph with average character length of sections

cols <- c("guest_access_length" = guest_access_color, 
          "desc_length" = description_color, 
          "other_things_length" = other_things_color, 
          "space_length" = space_color)

data %>% mutate(desc_length = nchar(Description), 
                space_length = nchar(Space), 
                guest_access_length = nchar(Guest_access), 
                other_things_length = nchar(Other_things_to_note)) %>%
  pivot_longer(cols = c(desc_length, space_length, guest_access_length, other_things_length), names_to = "section", values_to = "length") %>%
  group_by(section) %>%
  summarise(avg_length = mean(length, na.rm = TRUE)) %>%
  mutate(section = fct_reorder(section, avg_length)) %>% 
  ggplot(aes(x = avg_length, y = section, label = round(avg_length,0), fill = section)) +
  geom_col() + 
  scale_fill_manual(values = cols) +
  geom_label(fill = "White") +
  scale_y_discrete(labels = c("Guest access", "Description", "Other things to note ", "The space")) +
  labs(title = "Average length of Airbnb description sections",
       subtitle = "Space is the lengthiest section as hosts exceed 500 characters in the description",
       x = "Average character length",
       y = "Section") +
  theme(panel.background = element_rect(fill="white", linetype = "solid", color="black"),
        panel.grid.major = element_line( linetype = "blank"),
        panel.grid.minor = element_line( linetype = "blank"),
        plot.title = element_text(hjust = 0.5, size=18),
        plot.subtitle = element_text(hjust = 0.5, size=12),
        axis.text = element_text(size=10),
        axis.title = element_text(size=14),
        legend.position = "None") +
  theme(plot.margin = unit(c(1,1,1,1), "cm"))

# annotate text using udpipe

ud_model <- udpipe_download_model(language = "english")
ud_model <- udpipe_load_model(ud_model$file_model)

description_annotated <- udpipe_annotate(ud_model, x = data$Description, doc_id = data$id)
description_annotated <- as.data.frame(description_annotated)

other_things_annotated <- udpipe_annotate(ud_model, x = data$Other_things_to_note, doc_id = data$id)
other_things_annotated <- as.data.frame(other_things_annotated)

space_annotated <- udpipe_annotate(ud_model, x = data$Space, doc_id = data$id)
space_annotated <- as.data.frame(space_annotated)

guest_access_annotated <- udpipe_annotate(ud_model, x = data$Guest_access, doc_id = data$id)
guest_access_annotated <- as.data.frame(guest_access_annotated)


stop_words_custom <- stop_words %>% filter(!word == "room")

# function that will pre-process the the annotation for analysis including recoding common bigrams

prepare_annotation <- function(x) {
  
  find_ngrams <- x %>% 
    filter(!upos %in% c("PUNCT", "PROPN", "PART", "SYM", "CCONJ", "PART", "AUX", "X", "NUM")) %>%
    mutate(token = tolower(token)) %>%
    anti_join(stop_words_custom, by = c("token" = "word")) %>%
    mutate(bigram = txt_nextgram(lemma, n = 2))
  
  bigrams <- find_ngrams %>% count(bigram, sort = TRUE) %>% filter(n > 50) %>% select(bigram)
  bigrams <- unlist(bigrams)
  bigrams <- append(bigrams, c("check in", 
                               "check out"))   
  ngram <- rep(2, length(bigrams))
  x$term <- x$lemma
  x$term <- txt_recode_ngram(x$term, bigrams, ngram)
  
  x <- x %>% 
    mutate(term2=txt_recode_ngram(x$token, c("check - in", "check - out"), c(3,3))) %>%
    mutate(term = if_else((term2 == "check - in" | term2 == "check - out"), term2,term)) %>%
    select(-term2)
  
  x_prepared <- x %>% 
    filter(!upos %in% c("PUNCT", "PROPN", "PART", "SYM", "CCONJ", "PART", "AUX", "X", "NUM")) %>%
    mutate(token = tolower(token)) %>%
    anti_join(stop_words, by = c("token" = "word")) %>%
    filter(!is.na(term)) %>%
    mutate(term = tolower(term))
  
  x_prepared <- x_prepared %>% 
    mutate(term=str_replace(term,"check - in", "check-in")) %>%
    mutate(term=str_replace(term,"checkin", "check-in")) %>%
    mutate(term=str_replace(term,"check in", "check-in")) %>%
    mutate(term=str_replace(term,"check - out", "check-out")) %>%
    mutate(term=str_replace(term,"checkout", "check-out")) %>%
    mutate(term=str_replace(term,"check out", "check-out"))
    
    return 
  x_prepared 
  
}

# process each description section for network visualization 

description_annotated_prepared <- prepare_annotation(description_annotated)
space_annotated_prepared <- prepare_annotation(space_annotated)
guest_access_annotated_prepared <- prepare_annotation(guest_access_annotated)
other_things_annotated_prepared <- prepare_annotation(other_things_annotated)

description_annotated_prepared <- description_annotated_prepared %>% mutate(row = 1:n())
space_annotated_prepared <- space_annotated_prepared %>% mutate(row = 1:n())
guest_access_annotated_prepared <- guest_access_annotated_prepared %>% mutate(row = 1:n())
other_things_annotated_prepared <- other_things_annotated_prepared %>% mutate(row = 1:n())


# create network visualization function

visualize_bigrams <- function(annotated_prepared, color = "lightblue", n = 150, n2=124) {
  set.seed(n2)

  vert <- annotated_prepared %>%
    mutate(row2 = row+1) %>% 
    inner_join(annotated_prepared, by = c("row2" = "row")) %>%
    group_by(term.x, term.y) %>%
    count() %>%
    filter(term.x != term.y) %>%
    arrange(desc(n)) %>% 
    head(n) %>%
    gather(item, word, term.x, term.y) %>%
    group_by(word) %>% summarise(n = sum(n))  
  
  annotated_prepared %>%
    mutate(row2 = row+1) %>% 
    inner_join(annotated_prepared, by = c("row2" = "row")) %>%
    group_by(term.x, term.y) %>%
    count() %>%
    filter(term.x != term.y) %>%
    arrange(desc(n)) %>% 
    head(n) %>%
    graph_from_data_frame(vertices = vert) %>%
    ggraph(layout = "fr") +
    geom_edge_link(aes(edge_alpha = n), show.legend = FALSE, end_cap = circle(.05, 'inches')) +
    geom_node_point(aes(size = n), color = color, show.legend = FALSE) +
    scale_size(range = c(4,12)) +
    geom_node_text(aes(label = name), check_overlap=TRUE) +
    theme_void()
}

# create network visualization. Add annotations to description visualization 

description_circles <- data.frame(x0 = c(3, 8, 7.5, 15, 0,5), y0 = c(3, 0, 8, 4.5, 7,13.5), r = c(3,3,3,4.5,2,2.5))
description_annotation <- data.frame(x = c(2,10,7.5,16,-1,5.5), y = c(2,1.5,5.5,6,6,12), label = c("1","2","3","4","5","6"))

visualize_bigrams(description_annotated_prepared, color = description_color) +
  geom_circle(data = description_circles, aes(x0 = x0, y0 = y0, r = r), linetype = "dashed") + 
  geom_label(data=description_annotation, aes( x=x, y=y, label=label),color="white", fill="black", size=4 ,fontface="bold") +
  coord_fixed() +
  labs(title = "Network graph of Airbnb description") +
  theme(plot.title = element_text(hjust = 0.5, size=18)) 

visualize_bigrams(space_annotated_prepared, color = space_color) +
  labs(title = "Network graph of the space section of Airbnb description") +
  theme(plot.title = element_text(hjust = 0.5, size=18)) 


visualize_bigrams(guest_access_annotated_prepared, color = guest_access_color) +
  labs(title = "Network graph of the guest access section of Airbnb description") +
  theme(plot.title = element_text(hjust = 0.5, size=18)) 

visualize_bigrams(other_things_annotated_prepared, color = other_things_color) +
  labs(title = "Network graph of the other things section of Airbnb description") +
  theme(plot.title = element_text(hjust = 0.5, size=18)) 
