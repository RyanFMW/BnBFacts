library(tidyverse)
library(XML)
library(factoextra)
library(mice)
library(gt)


url <- "http://insideairbnb.com/get-the-data.html"

htmlParse("http://insideairbnb.com/get-the-data.html")

doc <- htmlParse(url)
links <- xpathSApply(doc, "//a/@href")
US_links <- links[grepl("united-states", links)]
US_links <- US_links[grepl("2020-02", US_links)]
US_links <- US_links[grepl("listings", US_links)]
US_links <- US_links[!grepl("visualisations", US_links)]
free(doc)

summary(US_links) #of files

GetMe <- paste(US_links, sep = "")

listings <- lapply(as.vector(GetMe), function(x)
{con <- gzcon(url(x))
txt <- readLines(con)
dat <- read.csv(textConnection(txt, encoding = "UTF-8"))}
)

listings.df <- do.call(rbind.data.frame,listings) 
listings.df$price <- as.numeric(gsub('[$,]', '', listings.df$price))
listings.df$host_response_rate <- as.numeric(gsub('[%]', '', listings.df$host_response_rate))
listings.df$host_acceptance_rate  <- as.numeric(gsub('[%]', '', listings.df$host_acceptance_rate))


listings.df <- listings.df %>% 
  mutate(max_listings=pmax(host_total_listings_count, host_listings_count, calculated_host_listings_count)) %>%
  mutate(price_per_accommodates=price/accommodates) %>% 
  mutate(days_occupied_estimate = if_else(minimum_nights_avg_ntm>3,minimum_nights_avg_ntm,3))
  
hosts <- listings.df %>% 
  filter(number_of_reviews>0, 
         minimum_nights_avg_ntm<=365, 
         number_of_reviews_ltm<365/days_occupied_estimate,
         price >0) %>%
  mutate(occupancy_estimate =(days_occupied_estimate*(number_of_reviews_ltm/0.5))/365) %>%
  mutate(occupancy_estimate=if_else(occupancy_estimate>0.7,0.7,occupancy_estimate)) %>%
  group_by(host_id) %>%
  summarise(
    max_listings=max(max_listings),
    avg_acceptance_rate=mean(host_acceptance_rate),
    avg_response_rate=mean(host_response_rate),
    avg_min_nights=mean(minimum_nights_avg_ntm),
    avg_occupancy_estimate=mean(occupancy_estimate),
    avg_price_per_accommodates=mean(price_per_accommodates),
    avg_accommodates=mean(accommodates),
    avg_reviews_ltm=mean(number_of_reviews_ltm))

hosts <- data.frame(hosts)
hosts.filtered <- hosts %>% filter(max_listings<30 & avg_price_per_accommodates<200)
rownames(hosts.filtered) <- hosts.filtered$host_id
hosts.filtered <- hosts.filtered[,-c(1,8,9)]
summary(hosts.filtered) 

hosts.imputed <- mice(hosts.filtered, m=5, maxit=5, method="pmm")

hosts.scaled <- scale(complete(hosts.imputed))

k.max <- 20
data <- hosts.scaled
wss <- sapply(2:k.max,
              function(k){kmeans(data, k, nstart=5, iter.max=25)$tot.withinss})

wss.plot <- data.frame(k=c(2:20), wss)

ggplot(wss.plot, aes(x=k, y=wss)) +
  geom_point() + 
  geom_line() +
  ggtitle("Scree Plot")

hosts.model <- kmeans(hosts.scaled, centers=7, nstart=5)

fviz_cluster(hosts.model, data = hosts.scaled,
             ellipse.type = "convex", 
             geom = "point",
             ggtheme = theme_bw())

hosts.filtered <- rownames_to_column(hosts.filtered,var="host_id")

hosts.filtered <- hosts.filtered %>% 
  mutate(cluster = hosts.model$cluster) %>%
  mutate(acceptance_imputed = is.na(avg_acceptance_rate)) %>%
  mutate(response_imputed = is.na(avg_response_rate))

hosts <- merge(x = hosts, y = hosts.filtered[, c("host_id", "cluster", "acceptance_imputed", "response_imputed")], by = "host_id", all.x=TRUE) 

hosts$cluster <- as.numeric(hosts$cluster)

hosts <- hosts %>% 
  mutate(cluster = if_else(max_listings>=30, 8, cluster)) %>%
  mutate(cluster = if_else(avg_price_per_accommodates>=200, 9, cluster))

host.clusters <- hosts %>% 
                      group_by(cluster) %>%
                      summarise(
                        count=n(),
                          avg_max_listings=mean(max_listings),
                          avg_acceptance_rate=mean(avg_acceptance_rate,na.rm=TRUE)/100,
                          avg_response_rate=mean(avg_response_rate,na.rm=TRUE)/100,
                          avg_min_nights=mean(avg_min_nights),
                          avg_occupancy_estimate=mean(avg_occupancy_estimate),
                          avg_price_per_accommodates=mean(avg_price_per_accommodates),
                          acceptance_imputed=sum(acceptance_imputed,na.rm=TRUE),
                          response_imputed=sum(response_imputed,na.rm = TRUE),
                          avg_accommodates=mean(avg_accommodates),
                          avg_reviews_ltm=mean(avg_reviews_ltm))

x <- data.frame(host.clusters)
write.table(x, "clipboard", sep="\t", row.names=FALSE)

host.clusters <- host.clusters %>% filter(!is.na(cluster))

host.clusters <- host.clusters %>% 
  mutate(value=(count*avg_max_listings*(365*avg_occupancy_estimate)*(avg_accommodates*avg_price_per_accommodates))) %>% 
  mutate(value_percent=value/sum(value)) %>%
  mutate(estimated_listing_count=count*avg_max_listings) %>%
  mutate(estimated_listing_count_percent=count*avg_max_listings/sum(count*avg_max_listings)) %>%
  mutate(host_percent=count/sum(count))

host.clusters$cluster_name <- c(
  "Part-time Selective",
  "Full-time Long-term",
  "Part-time Passive",
  "Full-time High-priced",
  "Part-time Active",
  "Full-time Active",
  "Full-time Mult-Unit",
  "Management Entity",
  "Full-time Luxury")

host.clusters %>% 
  arrange(desc(value)) %>%
  select(cluster_name, count, avg_max_listings, avg_acceptance_rate, avg_response_rate, avg_min_nights, avg_occupancy_estimate, avg_price_per_accommodates) %>%
  gt(rowname_col = "cluster_name") %>%
  fmt_number(vars(count), decimals=0, sep_mark = ",") %>%
  fmt_percent(vars(avg_acceptance_rate, avg_response_rate, avg_occupancy_estimate), decimals=0) %>%
  fmt_number(vars(avg_max_listings, avg_min_nights), decimals=2) %>%
  fmt_currency(vars(avg_price_per_accommodates)) %>%
  cols_label(cluster_name = "Cluster Name", 
             count = "Host Count", 
             avg_max_listings = "Avg. Listing Count", 
             avg_acceptance_rate = "Avg. Acceptance Rate", 
             avg_response_rate = "Avg. Response Rate", 
             avg_min_nights = "Avg. Minimum Nights", 
             avg_occupancy_estimate = "Avg. Occupancy Estimate", 
             avg_price_per_accommodates = "Avg. Price per Person") %>%
  data_color(vars(count, avg_max_listings, avg_min_nights, avg_occupancy_estimate, avg_price_per_accommodates), colors = scales::col_numeric(
    palette = c("transparent","powderblue"), 
    domain = NULL)) %>% 
  data_color(vars( avg_acceptance_rate, avg_response_rate), colors = scales::col_numeric(
    palette = c("powderblue","transparent"), 
    domain = NULL)) %>%
  tab_options(column_labels.font.weight="bold") %>%
  cols_width(vars(cluster_name) ~ px(200),
             vars(count, avg_max_listings, avg_acceptance_rate, avg_response_rate, avg_min_nights, avg_occupancy_estimate, avg_price_per_accommodates) ~ px(91))


host.clusters %>% 
  arrange(desc(value)) %>%
  mutate(value=value/1000) %>%
  select(cluster_name, count, host_percent, estimated_listing_count, estimated_listing_count_percent, value, value_percent) %>%
  gt(rowname_col = "cluster_name") %>%
  fmt_number(vars(count, estimated_listing_count), decimals=0, sep_mark = ",") %>%
  fmt_percent(vars(host_percent, estimated_listing_count_percent, value_percent), decimals=0) %>%
  fmt_currency(vars(value), decimals=0, suffixing=c("M")) %>%
  cols_label(cluster_name = "Cluster Name", 
             count = "Host Count", 
             host_percent = "% of Total Hosts", 
             estimated_listing_count = "Listing Count", 
             estimated_listing_count_percent = "% of Total Listings", 
             value = "Estimated Value", 
             value_percent = "% of Total Value") %>%
  data_color(vars(count, count, host_percent, estimated_listing_count, estimated_listing_count_percent, value, value_percent), colors = scales::col_numeric(
    palette = c("transparent","powderblue"), 
    domain = NULL)) %>%
  tab_options(column_labels.font.weight="bold") %>%
  cols_width(vars(cluster_name) ~ px(200),
             vars(count, count, host_percent, estimated_listing_count, estimated_listing_count_percent, value, value_percent) ~ px(110))

    
host.clusters$value  
glimpse(host.clusters)

host.clusters %>% 
  select(value) %>%
  gt() %>%
fmt_currency(vars(value), scale_by=0.000001, suffixing=TRUE)

host.clusters$count <- fmt_number(host.clusters$count, decimals=0,sep_mark = ",")
host.clusters$avg_max_listings <- digits(host.clusters$avg_max_listings,2)
host.clusters$avg_acceptance_rate <- percent(host.clusters$avg_acceptance_rate,0)
host.clusters$avg_response_rate <- percent(host.clusters$avg_response_rate,0)
host.clusters$avg_min_nights <- digits(host.clusters$avg_min_nights, 2)
host.clusters$avg_occupancy_estimate <- percent(host.clusters$avg_occupancy_estimate,0)
host.clusters$avg_price_per_accommodates <- currency(host.clusters$avg_price_per_accommodates)
host.clusters$value <- currency(host.clusters$value/100000)
host.clusters$value_percent <- percent(host.clusters$value_percent,0)
host.clusters$estimated_listing_count <- comma(host.clusters$estimated_listing_count,0)
host.clusters$estimated_listing_count_percent <- percent(host.clusters$estimated_listing_count_percent,0)
host.clusters$host_percent <- percent(host.clusters$host_percent ,0)



formattable(x, list(avg_min_nights = formatter("span",
                                    style = x ~ style(display = "block",
                                                      "border-radius" = "4px",
                                                      "padding-right" = "4px",
                                                      color = "white",
                                                      "background-color" = rgb(x/max(x), 0, 0)))))


format.values <-  formatter("span", style = x 
    style(display = "block",
          padding = "0 4px", 
          `border-radius` = "4px", 
          `background-color` = csscolor(matrix(as.integer(colorRamp(...)(normalize(as.numeric(x)))), 
                                               byrow=TRUE, dimnames=list(c("red","green","blue"), NULL), nrow=3))))



host.clusters %>% 
  select(cluster_name, count, avg_max_listings, avg_acceptance_rate, avg_response_rate, avg_min_nights, avg_occupancy_estimate, avg_price_per_accommodates) %>%
  formattable(col.names = c("Cluster Name", "Host Count", "Avg. Listing Count", "Avg. Acceptance Rate", "Avg. Response Rate", "Avg. Minimum Nights", "Avg. Occupancy Estimate", "Avg. Price per Person"),
  column_spec()            
    list(avg_acceptance_rate = color_tile("grey90", "powderblue"),
         avg_response_rate = color_tile("grey90", "powderblue", `border-radius` = "4px")))

host.clusters %>% 
  select(cluster_name, count, avg_max_listings, avg_acceptance_rate, avg_response_rate, avg_min_nights, avg_occupancy_estimate, avg_price_per_accommodates) %>%
  formattable(col.names = c("Cluster Name", "Host Count", "Avg. Listing Count", "Avg. Acceptance Rate", "Avg. Response Rate", "Avg. Minimum Nights", "Avg. Occupancy Estimate", "Avg. Price per Person"),
              list(area(col=2:8) ~ color_tile("transparent", "powderblue")))

colors()

x$cluster_name <- c(
"Full-time Active",
"Part-time Selective",
"Full-time High-priced",
"Full-time Mult-Unit",
"Part-time Active",
"Full-time Long-term",
"Part-time Passive",
"Management Entity",
"Full-time Luxury")

colnames(x) <- c("Cluster Number",
                 "Host Count",
                 "Avg. Listings", 
                 "Avg. Acceptance Rate", 
                 "Avg. Response Rate", 
                 "Avg. Minimum Nights", 
                 "Avg. Reviews LTM", 
                 "Avg. Price per Person", 
                 "Avg. Accommodates", 
                 "Value", 
                 "Cluster Name") 


rm(list=c("wss.plot", "listings.df.clustered", "x","hosts","hosts.filtered","hosts.imputed", 
          "hosts.model", "hosts.scaled", "listings.clustered", "host.clusters"))
