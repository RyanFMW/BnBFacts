library(tidyverse)
library(relaimpo)
library(broom)

# load data. data was scraped late January 2022
losangeles <- read_csv("losangeles.csv")
newyork <- read_csv("newyork.csv")
dallas <- read_csv("dallas.csv")
colorado_springs <- read_csv("colorado_springs.csv")
miami <- read_csv("miami.csv")

data <- rbind(losangeles, newyork, dallas, colorado_springs, miami)  
data$id = gsub("[^0-9]", "", data$url)

model <- lm(star_rating ~ rating_communication + rating_location + rating_value + rating_accuracy + rating_checkin + rating_cleanliness, data = data)

summary(model)
model_relimp <- calc.relimp(model, rela= TRUE)
model_relimp <- model_relimp$lmg %>% as.list %>% as.data.frame %>% tidyr::gather() %>% mutate(key = str_to_title(substring(key, 8)))
model_relimp 

model_relimp %>%
  ggplot(aes(x=reorder(key, -value), y = value, fill = key)) +
  geom_col() +
  scale_y_continuous(labels = scales::percent) +
  labs(title = "Relative Importance of Categories to Overall Rating",
       subtitle = "Accuracy, value and cleanliness are most important",
       x = "Rating category",
       y = "Relative importance") +
  theme(panel.background = element_rect(fill="white", linetype = "solid", color="black"),
        panel.grid.major = element_line( linetype = "blank"),
        panel.grid.minor = element_line( linetype = "blank"),
        plot.title = element_text(hjust = 0.5, size=18),
        plot.subtitle = element_text(hjust = 0.5, size=14),
        axis.text = element_text(size=10),
        axis.title = element_text(size=14),
        legend.position="none") 

model2 <- lm(rating_value ~ rating_communication + rating_location +  rating_accuracy + rating_checkin + rating_cleanliness, data = data)

summary(model2)
model2_relimp <- calc.relimp(model2, rela= TRUE)
model2_relimp <- model2_relimp$lmg %>% as.list %>% as.data.frame %>% tidyr::gather() %>% mutate(key = str_to_title(substring(key, 8)))

model2_relimp %>%
  ggplot(aes(x=reorder(key, -value), y = value, fill = key)) +
  geom_col() +
  scale_y_continuous(labels = scales::percent) +
  labs(title = "Relative Importance of Other Categories to Value Rating",
       subtitle = "Accuracy is very important in determining the value rating",
       x = "Rating category",
       y = "Relative importance") +
  geom_text(aes(label = scales::percent(value)), vjust = -0.5) +
  theme(panel.background = element_rect(fill="white", linetype = "solid", color="black"),
        panel.grid.major = element_line( linetype = "blank"),
        panel.grid.minor = element_line( linetype = "blank"),
        plot.title = element_text(hjust = 0.5, size=18),
        plot.subtitle = element_text(hjust = 0.5, size=14),
        axis.text = element_text(size=10),
        axis.title = element_text(size=14),
        legend.position="none") 
  
