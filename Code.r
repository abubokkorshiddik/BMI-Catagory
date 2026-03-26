

library(readxl)
library(tidyverse)
library(openxlsx)  # to save Excel

# -----------------------
# Load data
# -----------------------
file_path <- "J:/Research/Research/WorldBMI/ABSDATA/BMIDATA.xlsx"
mydata <- read_excel(file_path, sheet = "data")

# -----------------------
# Select relevant columns
# -----------------------
bmi_data <- mydata %>%
  select(Country, Year, obesity, overweight, underweight)

# -----------------------
# Convert to long format
# -----------------------
bmi_long <- bmi_data %>%
  pivot_longer(cols = c(obesity, overweight, underweight),
               names_to = "Category",
               values_to = "Percent")

# -----------------------
# Summarize global average per year
# -----------------------
bmi_summary <- bmi_long %>%
  group_by(Year, Category) %>%
  summarise(GlobalAvg = mean(Percent, na.rm = TRUE), .groups = 'drop')

# -----------------------
# Save summarized data to Excel
# -----------------------
write.xlsx(bmi_summary, "J:/Research/Research/WorldBMI/ABSDATA/BMI_GlobalSummary.xlsx")

# -----------------------
# Plot bar chart
# -----------------------
plot <- ggplot(bmi_summary, aes(x = factor(Year), y = GlobalAvg, fill = Category)) +
  geom_col(position = "dodge") +
  labs(title = "Global BMI category prevalence (1990-2022)",
       x = "Year",
       y = "Percentage of population",
       fill = "BMI Category") +
  scale_fill_manual(values = c("obesity" = "#E41A1C",      # red
                               "overweight" = "#377EB8",  # blue
                               "underweight" = "#4DAF4A")) + # green
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1),
        plot.title = element_text(face = "bold", size = 14))

# -----------------------
# Save plot to file
# -----------------------
ggsave("J:/Research/Research/WorldBMI/ABSDATA/BMI_GlobalBarPlot.png",
       plot = plot,
       width = 16, height = 8, dpi = 300)



library(readxl)
library(tidyverse)
library(openxlsx)

# -----------------------
# Load data
# -----------------------
file_path <- "J:/Research/Research/WorldBMI/ABSDATA/BMIDATA.xlsx"
mydata <- read_excel(file_path, sheet = "data")

# -----------------------
# Select BMI variables
# -----------------------
bmi_data <- mydata %>%
  select(Country, Year, obesity, overweight, underweight)

# -----------------------
# Convert to long format
# -----------------------
bmi_long <- bmi_data %>%
  pivot_longer(cols = c(obesity, overweight, underweight),
               names_to = "Category",
               values_to = "Percent")

# -----------------------
# Calculate global mean and SD for each year
# -----------------------
bmi_summary <- bmi_long %>%
  group_by(Year, Category) %>%
  summarise(
    Mean = mean(Percent, na.rm = TRUE),
    SD = sd(Percent, na.rm = TRUE),
    Lower = Mean - SD,
    Upper = Mean + SD,
    .groups = "drop"
  )

# -----------------------
# Save summarized values
# -----------------------
write.xlsx(bmi_summary,
           "J:/Research/Research/WorldBMI/ABSDATA/BMI_Global_Mean_SD.xlsx")

# -----------------------
# Create bar plot with SD interval
# -----------------------
p <- ggplot(bmi_summary,
            aes(x = factor(Year), y = Mean, fill = Category)) +
  geom_bar(stat = "identity", position = position_dodge(width = 0.8)) +
  geom_errorbar(aes(ymin = Lower, ymax = Upper),
                width = 0.25,
                position = position_dodge(width = 0.8)) +
  labs(
    title = "Global BMI category prevalence with mean and SD interval (1990 to 2022)",
    x = "Year",
    y = "Global average prevalence (%)",
    fill = ""
  ) +
  scale_fill_manual(values = c(
    "obesity" = "#D73027",
    "overweight" = "#4575B4",
    "underweight" = "#1A9850"
  )) +
  theme_minimal() +
  theme(
    axis.text.x = element_text(angle = 90, size = 8),
    plot.title = element_text(size = 14, face = "bold"),
    legend.position = "bottom"
  )
p
# -----------------------
# Save plot
# -----------------------
ggsave(
  "J:/Research/Research/WorldBMI/ABSDATA/BMI_Global_Mean_SD_BarPlot.png",
  plot = p,
  width = 18,
  height = 8,
  dpi = 300
)








library(readxl)
library(tidyverse)
library(sf)
library(rnaturalearth)
library(rnaturalearthdata)
library(ggspatial)
library(openxlsx)
library(viridis)

# Load dataset
file_path <- "J:/Research/Research/WorldBMI/ABSDATA/BMIDATA.xlsx"
mydata <- read_excel(file_path, sheet = "data")

# Convert BMI variables to long format
bmi_long <- mydata %>%
  select(Country, Year, obesity, overweight, underweight) %>%
  pivot_longer(cols = c(obesity, overweight, underweight),
               names_to = "Category",
               values_to = "Percent")

# Country mean prevalence
bmi_country_mean <- bmi_long %>%
  group_by(Country, Category) %>%
  summarise(Mean = mean(Percent, na.rm = TRUE), .groups = "drop") %>%
  filter(!is.na(Mean))

# World map
world <- ne_countries(scale = "medium", returnclass = "sf") %>%
  filter(name != "Antarctica")

world_bmi <- world %>%
  left_join(bmi_country_mean, by = c("name" = "Country"))

# Map creation function
create_map <- function(cat_name){
  
  map_data <- world_bmi %>%
    filter(Category == cat_name & !is.na(Mean))
  
  ggplot(map_data) +
    geom_sf(aes(fill = Mean),
            color = "black",
            size = 0.15) +
    scale_fill_viridis(option = "turbo",
                       name = "Prevalence (%)") +
    labs(
      title = paste("Global distribution of", cat_name, "prevalence")
    ) +
    annotation_scale(location = "bl", width_hint = 0.35) +
    annotation_north_arrow(location = "tr",
                           which_north = "true",
                           style = north_arrow_minimal) +
    theme_void() +
    theme(
      legend.position = "bottom",
      plot.title = element_text(size = 16, face = "bold", hjust = 0.5)
    )
}

# Create maps
map_obesity <- create_map("obesity")
map_overweight <- create_map("overweight")
map_underweight <- create_map("underweight")

# Save maps
ggsave("J:/Research/Research/WorldBMI/ABSDATA/map_obesity.png",
       map_obesity, width = 12, height = 7, dpi = 300)

ggsave("J:/Research/Research/WorldBMI/ABSDATA/map_overweight.png",
       map_overweight, width = 12, height = 7, dpi = 300)

ggsave("J:/Research/Research/WorldBMI/ABSDATA/map_underweight.png",
       map_underweight, width = 12, height = 7, dpi = 300)














library(readxl)
library(tidyverse)
library(openxlsx)

# Load dataset
file_path <- "J:/Research/Research/WorldBMI/ABSDATA/BMIDATA.xlsx"
mydata <- read_excel(file_path, sheet = "data")

# Convert to long format
bmi_long <- mydata %>%
  select(Country, Year, obesity, overweight, underweight) %>%
  pivot_longer(cols = c(obesity, overweight, underweight),
               names_to = "Category",
               values_to = "Percent")

# Function to get top and bottom 20 countries per category with full summary
get_country_summary <- function(cat_name){
  
  # Filter for category
  cat_data <- bmi_long %>% filter(Category == cat_name)
  
  # Country mean for ranking
  country_mean <- cat_data %>%
    group_by(Country) %>%
    summarise(Mean = mean(Percent, na.rm = TRUE), .groups = "drop") %>%
    filter(!is.na(Mean))
  
  # Top 20 highest and lowest
  top20_high <- country_mean %>%
    arrange(desc(Mean)) %>% slice_head(n = 20)
  
  top20_low <- country_mean %>%
    arrange(Mean) %>% slice_head(n = 20)
  
  # Combine
  selected_countries <- bind_rows(top20_high, top20_low)
  
  # Calculate full summary for each country
  country_summary <- cat_data %>%
    filter(Country %in% selected_countries$Country) %>%
    group_by(Country) %>%
    summarise(
      Min = min(Percent, na.rm = TRUE),
      Q1 = quantile(Percent, 0.25, na.rm = TRUE),
      Median = median(Percent, na.rm = TRUE),
      Q3 = quantile(Percent, 0.75, na.rm = TRUE),
      Max = max(Percent, na.rm = TRUE),
      Mean_value = mean(Percent, na.rm = TRUE),
      SD = sd(Percent, na.rm = TRUE),
      Mean_SD = paste0(round(Mean_value,2)," ± ",round(SD,2)),
      .groups = "drop"
    )
  
  return(country_summary)
}

# Generate summaries for all three categories
summary_obesity <- get_country_summary("obesity")
summary_overweight <- get_country_summary("overweight")
summary_underweight <- get_country_summary("underweight")

# Save to Excel with separate sheets for each category
wb <- createWorkbook()

addWorksheet(wb, "Obesity")
writeData(wb, "Obesity", summary_obesity)

addWorksheet(wb, "Overweight")
writeData(wb, "Overweight", summary_overweight)

addWorksheet(wb, "Underweight")
writeData(wb, "Underweight", summary_underweight)

saveWorkbook(wb,
             "J:/Research/Research/WorldBMI/ABSDATA/BMI_Country_TopBottom_Summary.xlsx",
             overwrite = TRUE)

# Print example outputs
print(summary_obesity)
print(summary_overweight)
print(summary_underweight)





library(readxl)
library(tidyverse)
library(openxlsx)

# Load dataset
file_path <- "J:/Research/Research/WorldBMI/ABSDATA/BMIDATA.xlsx"
mydata <- read_excel(file_path, sheet = "data")

# Convert to long format
bmi_long <- mydata %>%
  select(Country, Year, obesity, overweight, underweight) %>%
  pivot_longer(cols = c(obesity, overweight, underweight),
               names_to = "Category",
               values_to = "Percent")

# Function to calculate year-wise summary for a category
yearwise_summary <- function(cat_name){
  bmi_long %>%
    filter(Category == cat_name) %>%
    group_by(Year) %>%
    summarise(
      Min = min(Percent, na.rm = TRUE),
      Q1 = quantile(Percent, 0.25, na.rm = TRUE),
      Median = median(Percent, na.rm = TRUE),
      Q3 = quantile(Percent, 0.75, na.rm = TRUE),
      Max = max(Percent, na.rm = TRUE),
      Mean_value = mean(Percent, na.rm = TRUE),
      SD = sd(Percent, na.rm = TRUE),
      Mean_SD = paste0(round(Mean_value,2)," ± ", round(SD,2)),
      .groups = "drop"
    )
}

# Generate year-wise summaries for all categories
yearwise_obesity <- yearwise_summary("obesity")
yearwise_overweight <- yearwise_summary("overweight")
yearwise_underweight <- yearwise_summary("underweight")

# Save to Excel with separate sheets for each category
wb <- createWorkbook()

addWorksheet(wb, "Obesity")
writeData(wb, "Obesity", yearwise_obesity)

addWorksheet(wb, "Overweight")
writeData(wb, "Overweight", yearwise_overweight)

addWorksheet(wb, "Underweight")
writeData(wb, "Underweight", yearwise_underweight)

saveWorkbook(wb,
             "J:/Research/Research/WorldBMI/ABSDATA/BMI_Yearwise_Summary.xlsx",
             overwrite = TRUE)

# Print example outputs
print(yearwise_obesity)
print(yearwise_overweight)
print(yearwise_underweight)













library(ggplot2)
library(readxl)
library(dplyr)

# Load sheets
file_path <- "J:/Research/Research/WorldBMI/ABSDATA/Sumarry/BMI_Yearwise_Summary.xlsx"
obesity <- read_excel(file_path, sheet = "Obesity") %>% mutate(Type = "Obesity")
overweight <- read_excel(file_path, sheet = "Overweight") %>% mutate(Type = "Overweight")
underweight <- read_excel(file_path, sheet = "Underweight") %>% mutate(Type = "Underweight")

# Combine datasets
combined <- bind_rows(obesity, overweight, underweight)

# Publication-ready boxplot with larger fonts
p <- ggplot(combined, aes(x = factor(Year), fill = Type)) +
  geom_boxplot(aes(lower = Q1, upper = Q3, middle = Median, ymin = Min, ymax = Max),
               stat = "identity",
               position = position_dodge(width = 0.7),
               width = 0.5,
               alpha = 1,
               color = "gray20") +
  scale_fill_manual(values = c("Obesity" = "#8B0000",
                               "Overweight" = "#FF8C00",
                               "Underweight" = "#00008B")) +
  labs(title = "A",
       x = "Year",
       y = "Prevalence (%)",
       fill = "BMI category") +
  theme_minimal() +
  theme(text = element_text(family = "Times New Roman"),
        axis.text.x = element_text(angle = 60, hjust = 1, size = 18),  # increased
        axis.text.y = element_text(size = 23),                           # increased
        axis.title = element_text(size = 23, face = "bold"),            # increased
        legend.position = "bottom",
        legend.title = element_text(size = 23),
        legend.text = element_text(size = 23),
        plot.title = element_text(size = 23, face = "bold"),
        panel.grid.major.x = element_blank(),
        panel.grid.minor.x = element_blank())

# Show plot
#print(p)

# Save high-resolution PNG
ggsave("J:/Research/Research/WorldBMI/ABSDATA/Sumarry/Global_BMI_Boxplot_Publication_LargeFont.png",
       plot = p, width = 18, height = 5, dpi = 600, type = "cairo")









library(readxl)
library(dplyr)

# File paths
obesity_file <- "J:/Research/Research/WorldBMI/ABSDATA/GiGlobal/GiStar_Hotspot_Obesity_Overall.xlsx"
overweight_file <- "J:/Research/Research/WorldBMI/ABSDATA/GiGlobal/GiStar_Hotspot_overweight_Overall.xlsx"
underweight_file <- "J:/Research/Research/WorldBMI/ABSDATA/GiGlobal/GiStar_Hotspot_underweight_Overall.xlsx"

# Read sheets (assuming the first sheet in each file)
obesity <- read_excel(obesity_file, sheet = 1) %>% mutate(Type = "Obesity")
overweight <- read_excel(overweight_file, sheet = 1) %>% mutate(Type = "Overweight")
underweight <- read_excel(underweight_file, sheet = 1) %>% mutate(Type = "Underweight")

# Combine all three datasets
combined <- bind_rows(obesity, overweight, underweight)

# Check the first few rows
head(combined)









library(ggplot2)
library(dplyr)
library(readxl)
library(rnaturalearth)
library(rnaturalearthdata)
library(sf)

# File paths
folder_path <- "J:/Research/Research/WorldBMI/ABSDATA/GiGlobal/"

obesity_file <- paste0(folder_path, "GiStar_Hotspot_Obesity_Overall.xlsx")
overweight_file <- paste0(folder_path, "GiStar_Hotspot_overweight_Overall.xlsx")
underweight_file <- paste0(folder_path, "GiStar_Hotspot_underweight_Overall.xlsx")

# Load world map without Antarctica
world <- ne_countries(scale = "medium", returnclass = "sf") %>%
  filter(continent != "Antarctica")

# Standardize country names
standardize_names <- function(df){
  df %>%
    mutate(Country = case_when(
      Country == "United States" ~ "United States of America",
      Country == "Russia" ~ "Russian Federation",
      Country == "South Korea" ~ "Republic of Korea",
      Country == "North Korea" ~ "Dem. Rep. Korea",
      Country == "Czech Republic" ~ "Czechia",
      Country == "Iran" ~ "Iran (Islamic Republic of)",
      TRUE ~ Country
    ))
}

# Read and prepare dataset
read_and_prepare <- function(file_path, type_name){
  read_excel(file_path) %>%
    mutate(Type = type_name) %>%
    standardize_names() %>%
    mutate(GiZ_plot = GiZscore)  # use all GiZscore
}











library(readxl)
library(dplyr)
library(tidyr)
library(ggplot2)
library(sf)
library(rnaturalearth)
library(rnaturalearthdata)
library(patchwork)

# ===============================
# read Excel file
# ===============================
file_path <- "J:/Research/Research/WorldBMI/ABSDATA/XGeoAI/IG_GeoAI_Results.xlsx"
Country_IG <- read_excel(file_path, sheet = "Country_IG")

# ===============================
# standardize country names
# ===============================
standardize_names <- function(df){
  df %>%
    mutate(Country = case_when(
      Country == "United States" ~ "United States of America",
      Country == "Russia" ~ "Russian Federation",
      Country == "South Korea" ~ "Republic of Korea",
      Country == "North Korea" ~ "Dem. Rep. Korea",
      Country == "Czech Republic" ~ "Czechia",
      Country == "Iran" ~ "Iran (Islamic Republic of)",
      TRUE ~ Country
    ))
}

Country_IG <- standardize_names(Country_IG)

# ===============================
# select top 5 features globally
# ===============================
global_ig <- read_excel(file_path, sheet = "Global_IG")
top5_features <- global_ig %>%
  arrange(desc(Mean_IG)) %>%
  slice(1:5) %>%
  pull(Feature)

# ===============================
# get world map
# ===============================
world <- ne_countries(scale="medium", returnclass="sf") %>%
  filter(name != "Antarctica")

# ===============================
# create maps for each top feature
# ===============================
map_list <- list()
for (feat in top5_features){
  
  # prepare data
  map_data <- Country_IG %>%
    select(Country, all_of(feat)) %>%
    rename(IG_Value = all_of(feat))
  
  # join with world shapefile
  world_map <- left_join(world, map_data, by=c("name"="Country"))
  
  # create map
  p <- ggplot(world_map) +
    geom_sf(aes(fill=IG_Value), color="black", size=0.1) +
    scale_fill_gradientn(colors=c("#D9E2F3","#7B9CCB","#4B6FA5","#2A4880","#0F2455"),
                         na.value="gray90",
                         name=paste0(feat," IG")) +
    theme_minimal() +
    labs(title=paste0("Country-level IG: ", feat)) +
    theme(axis.text=element_blank(),
          axis.ticks=element_blank(),
          panel.grid=element_blank(),
          plot.title=element_text(size=12, face="bold"),
          legend.position="bottom")
  
  map_list[[feat]] <- p
}

# ===============================
# combine all 5 maps in 2 columns
# ===============================
combined_maps <- wrap_plots(map_list, ncol=2)
combined_maps

# ===============================
# save the combined maps
# ===============================
ggsave("J:/Research/Research/WorldBMI/ABSDATA/XGeoAI/Country_IG_Top5_Features.png",
       combined_maps, width=14, height=12, dpi=600)
obesity <- read_and_prepare(obesity_file, "Obesity")
overweight <- read_and_prepare(overweight_file, "Overweight")
underweight <- read_and_prepare(underweight_file, "Underweight")

# Function to plot and save map
plot_and_save_giz <- function(df, label, save_name){
  map_data <- left_join(world, df, by = c("name" = "Country"))
  
  p <- ggplot(map_data) +
    geom_sf(aes(fill = GiZ_plot), color = "gray40", size = 0.2) +
    scale_fill_gradient2(
      low = "#00008B",   # coldspot
      mid = "white",     # zero
      high = "#FF0000",  # hotspot
      na.value = "white",
      midpoint = 0,
      name = "GiZscore"
    ) +
    labs(title = label,
         subtitle = paste0("Gi* (", unique(df$Type), ")")) +
    theme_minimal(base_size = 14) +
    theme(axis.text = element_blank(),
          axis.title = element_blank(),
          panel.grid = element_blank(),
          plot.title = element_text(size = 18, face = "bold"),
          plot.subtitle = element_text(size = 14, face = "italic"),
          legend.position = "bottom")
  
  print(p)
  
  # Save
  ggsave(paste0(folder_path, save_name), plot = p, width = 14, height = 8, dpi = 600)
}

# Plot and save all three
plot_and_save_giz(obesity, "A", "Obesity_GiZscore_Map.png")
plot_and_save_giz(overweight, "B", "Overweight_GiZscore_Map.png")
plot_and_save_giz(underweight, "C", "Underweight_GiZscore_Map.png")










library(readxl)
library(dplyr)
library(ggplot2)
library(rnaturalearth)
library(rnaturalearthdata)
library(sf)
library(ggspatial)  # for scale and north arrow

# -----------------------
# Load data
# -----------------------
file_path <- "J:/Research/Research/WorldBMI/ABSDATA/BMIDATA.xlsx"
mydata <- read_excel(file_path, sheet = "data")

# -----------------------
# Summarize mean ± SD by country
# -----------------------
bmi_summary <- mydata %>%
  group_by(Country) %>%
  summarise(
    obesity_mean = mean(obesity, na.rm = TRUE),
    obesity_sd = sd(obesity, na.rm = TRUE),
    overweight_mean = mean(overweight, na.rm = TRUE),
    overweight_sd = sd(overweight, na.rm = TRUE),
    underweight_mean = mean(underweight, na.rm = TRUE),
    underweight_sd = sd(underweight, na.rm = TRUE)
  ) %>%
  ungroup()

# -----------------------
# Load world map without Antarctica
# -----------------------
world <- ne_countries(scale = "medium", returnclass = "sf") %>%
  filter(continent != "Antarctica")

# -----------------------
# Standardize country names
# -----------------------
bmi_summary$Country[bmi_summary$Country == "United States"] <- "United States of America"
bmi_summary$Country[bmi_summary$Country == "Russia"] <- "Russian Federation"
bmi_summary$Country[bmi_summary$Country == "South Korea"] <- "Republic of Korea"
bmi_summary$Country[bmi_summary$Country == "North Korea"] <- "Dem. Rep. Korea"
bmi_summary$Country[bmi_summary$Country == "Czech Republic"] <- "Czechia"
bmi_summary$Country[bmi_summary$Country == "Iran"] <- "Iran (Islamic Republic of)"

# -----------------------
# Prepare map data for each BMI category
# -----------------------
map_obesity <- left_join(world, bmi_summary, by = c("name" = "Country"))
map_overweight <- left_join(world, bmi_summary, by = c("name" = "Country"))
map_underweight <- left_join(world, bmi_summary, by = c("name" = "Country"))

# -----------------------
# Plot Obesity mean ± SD
# -----------------------
pB <- ggplot(map_obesity) +
  geom_sf(aes(fill = obesity_mean), color = "black", size = 0.2) +
  scale_fill_gradient2(
    low = "#00008B", mid = "skyblue", high = "#FF0000",
    midpoint = mean(bmi_summary$obesity_mean, na.rm = TRUE),
    na.value = "white",
    name = "Obesity (%)"
  ) +
  labs(title = "B") +
  theme_minimal(base_size = 14) +
  annotation_scale(location = "bl", width_hint = 0.4) +
  annotation_north_arrow(location = "tr", which_north = "true", style = north_arrow_fancy_orienteering()) +
  theme(axis.text = element_blank(),
        axis.title = element_blank(),
        panel.grid = element_blank(),
        plot.title = element_text(size = 18, face = "bold"),
        legend.position = "bottom")
#print(pB)

ggsave("J:/Research/Research/WorldBMI/ABSDATA/Obesity_Mean_SD_Map.png", plot = pB, width = 12, height = 6, dpi = 600)

# -----------------------
# Plot Overweight mean ± SD
# -----------------------
pC <- ggplot(map_overweight) +
  geom_sf(aes(fill = overweight_mean), color = "black", size = 0.2) +
  scale_fill_gradient2(
    low = "#00008B", mid = "skyblue", high = "#FF0000",
    midpoint = mean(bmi_summary$overweight_mean, na.rm = TRUE),
    na.value = "white",
    name = "Overweight (%)"
  ) +
  labs(title = "C") +
  theme_minimal(base_size = 14) +
  annotation_scale(location = "bl", width_hint = 0.4) +
  annotation_north_arrow(location = "tr", which_north = "true", style = north_arrow_fancy_orienteering()) +
  theme(axis.text = element_blank(),
        axis.title = element_blank(),
        panel.grid = element_blank(),
        plot.title = element_text(size = 18, face = "bold"),
        legend.position = "bottom")
#print(pC)

ggsave("J:/Research/Research/WorldBMI/ABSDATA/Overweight_Mean_SD_Map.png", plot = pC, width = 12, height = 6, dpi = 600)

# -----------------------
# Plot Underweight mean ± SD
# -----------------------
pD <- ggplot(map_underweight) +
  geom_sf(aes(fill = underweight_mean), color = "black", size = 0.2) +
  scale_fill_gradient2(
    low = "#00008B", mid = "skyblue", high = "#FF0000",
    midpoint = mean(bmi_summary$underweight_mean, na.rm = TRUE),
    na.value = "white",
    name = "Underweight (%)"
  ) +
  labs(title = "D") +
  theme_minimal(base_size = 14) +
  annotation_scale(location = "bl", width_hint = 0.4) +
  annotation_north_arrow(location = "tr", which_north = "true", style = north_arrow_fancy_orienteering()) +
  theme(axis.text = element_blank(),
        axis.title = element_blank(),
        panel.grid = element_blank(),
        plot.title = element_text(size = 18, face = "bold"),
        legend.position = "bottom")
#print(pD)

ggsave("J:/Research/Research/WorldBMI/ABSDATA/Underweight_Mean_SD_Map.png", plot = pD, width = 12, height = 6, dpi = 600)






library(readxl)
library(dplyr)
library(ggplot2)

# ===============================
# file paths
# ===============================
obesity_path <- "J:/Research/Research/WorldBMI/ABSDATA/corr/SpatioTemporalWeightedCorr_Bootstrap.xlsx"
overweight_path <- "J:/Research/Research/WorldBMI/ABSDATA/corr/overweight_SpatioTemporalWeightedCorr_Bootstrap.xlsx"
underweight_path <- "J:/Research/Research/WorldBMI/ABSDATA/corr/underweightt_SpatioTemporalWeightedCorr_Bootstrap.xlsx"

save_path <- "J:/Research/Research/WorldBMI/ABSDATA/corr/"

# ===============================
# read data and add category
# ===============================
obesity_df <- read_excel(obesity_path) %>% mutate(category = "Obesity")
overweight_df <- read_excel(overweight_path) %>% mutate(category = "Overweight")
underweight_df <- read_excel(underweight_path) %>% mutate(category = "Underweight")

# ===============================
# combine data
# ===============================
combined_df <- bind_rows(obesity_df, overweight_df, underweight_df)

# ===============================
# grouped bar plot with legend at bottom
# ===============================
p <- ggplot(combined_df, aes(x = Feature, y = WeightedCorr, fill = category)) +
  geom_bar(stat = "identity", position = position_dodge(width = 0.8)) +
  theme_minimal() +
  labs(title = "",
       x = "Features",
       y = "Weighted correlation",
       fill = "BMI Category") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1),
        legend.position = "bottom") +
  scale_fill_manual(values = c("Obesity" = "#1B365D",
                               "Overweight" = "#8A1538",
                               "Underweight" = "#4A4A4A"))

# ===============================
# save plot
# ===============================
ggsave(paste0(save_path, "BMI_Correlation_3in1_bar_legend_bottom.png"), p, width = 10, height = 6)










library(ggplot2)
library(dplyr)
library(tidyr)

# ===============================
# create dataset manually with CI
# ===============================
# Obesity
obesity_df <- tibble(
  Model = rep(c("ConvLSTM","GAT","GWNN","STGNN"), each = 2),
  Type = rep(c("Train","Test"), times = 4),
  RMSE = c(0.2044,0.3474, 0.2786,0.4917, 0.1868,0.4711, 0.1831,0.2748),
  RMSE_lower = c(0.2004,0.3181, 0.2731,0.4662, 0.1833,0.4468, 0.1816,0.2596),
  RMSE_upper = c(0.2085,0.3767, 0.2842,0.5171, 0.1902,0.4955, 0.1846,0.2900),
  MAE  = c(0.1437,0.2764, 0.1779,0.3762, 0.1293,0.3604, 0.1214,0.1994),
  MAE_lower = c(0.1396,0.2471,0.1724,0.3508,0.1258,0.3361,0.1199,0.1842),
  MAE_upper = c(0.1477,0.3057,0.1834,0.4016,0.1328,0.3848,0.1329,0.2146),
  R2   = c(0.9347,0.6914,0.8867,0.4027,0.9491,0.4516,0.9511,0.8134),
  Adj_R2 = c(0.9344,0.6528,0.8862,0.3798,0.9489,0.4305,0.951,0.8062)
)

# Overweight
overweight_df <- tibble(
  Model = rep(c("ConvLSTM","GAT","GWNN","STGNN"), each = 2),
  Type = rep(c("Train","Test"), times = 4),
  RMSE = c(0.2450,0.2771,0.1956,0.3082,0.1761,0.4927,0.1390,0.1648),
  RMSE_lower = c(0.2402,0.2536,0.1919,0.2908,0.1534,0.4650,0.1379,0.1561),
  RMSE_upper = c(0.2498,0.3005,0.1993,0.3256,0.1788,0.5203,0.1401,0.1735),
  MAE  = c(0.1742,0.2199,0.1336,0.2192,0.1171,0.3517,0.1019,0.1244),
  MAE_lower = c(0.1694,0.1964,0.1299,0.2018,0.1124,0.3240,0.1008,0.1157),
  MAE_upper = c(0.1790,0.2434,0.1373,0.2366,0.1178,0.3794,0.1030,0.1331),
  R2   = c(0.815,0.5305,0.8936,0.4428,0.9222,0.11,0.9463,0.8407),
  Adj_R2 = c(0.8142,0.4718,0.8932,0.4215,0.922,0.11,0.9463,0.8346)
)

# Underweight
underweight_df <- tibble(
  Model = rep(c("ConvLSTM","GAT","GWNN","STGNN"), each = 2),
  Type = rep(c("Train","Test"), times = 4),
  RMSE = c(0.1539,0.2836,0.2616,0.5523,0.1441,0.5357,0.1141,0.2150),
  RMSE_lower = c(0.1509,0.2578,0.2564,0.5179,0.1416,0.5053,0.1132,0.2036),
  RMSE_upper = c(0.1569,0.3093,0.2668,0.5868,0.1466,0.5662,0.1150,0.2264),
  MAE  = c(0.1098,0.2149,0.1674,0.3475,0.1064,0.3782,0.0845,0.1617),
  MAE_lower = c(0.1068,0.1891,0.1622,0.3130,0.1039,0.3477,0.0836,0.1504),
  MAE_upper = c(0.1128,0.2406,0.1726,0.3820,0.1089,0.4087,0.0853,0.1731),
  R2   = c(0.9585,0.8201,0.8826,0.3299,0.9643,0.3695,0.9776,0.8985),
  Adj_R2 = c(0.9584,0.7976,0.8821,0.3042,0.9642,0.3453,0.9776,0.8946)
)

library(ggplot2)
library(dplyr)
library(tidyr)
library(cowplot)  # for get_legend()

# reshape function
reshape_long <- function(df){
  df %>%
    pivot_longer(cols = c(RMSE, MAE, R2, Adj_R2),
                 names_to = "Metric", values_to = "Value")
}

obesity_long <- reshape_long(obesity_df)
overweight_long <- reshape_long(overweight_df)
underweight_long <- reshape_long(underweight_df)

# professional colors
model_colors <- c(
  "ConvLSTM"="#1F77B4",
  "GAT"="#2CA02C",
  "GWNN"="#9467BD",
  "STGNN"="#D62728"
)

# single plot function
plot_metric_train_test <- function(df_long, title_letter){
  p <- ggplot(df_long, aes(x=Metric, y=Value, fill=Model)) +
    geom_bar(stat="identity", position=position_dodge(width=0.8), color="black") +
    geom_errorbar(data=df_long %>% filter(Metric %in% c("RMSE","MAE")),
                  aes(ymin = ifelse(Metric=="RMSE", RMSE_lower, MAE_lower),
                      ymax = ifelse(Metric=="RMSE", RMSE_upper, MAE_upper)),
                  width=0.2, position=position_dodge(0.8)) +
    facet_wrap(~Type, nrow=1) +   # separate Train and Test in horizontal facets
    scale_fill_manual(values=model_colors) +
    labs(x="Metric", y="Value", fill="Model") +
    theme_minimal() +
    theme(plot.title = element_text(size=16, face="bold"),
          axis.text.x = element_text(angle=45, hjust=1),
          legend.position = "bottom",
          legend.title = element_text(face="bold"))
  
  # extract legend
  legend <- get_legend(p + guides(fill=guide_legend(nrow=1)))
  
  # remove legend from main plot
  p <- p + theme(legend.position="none")
  
  # combine plot and legend
  final <- plot_grid(p, legend, ncol=1, rel_heights=c(1,0.1))
  
  # add title letter
  final <- ggdraw() + 
    draw_plot(final) + 
    draw_label(title_letter, x=0.04, y=0.96, hjust=0, fontface="bold", size=18)
  
  return(final)
}

# create plots
pA <- plot_metric_train_test(obesity_long, "A")
pB <- plot_metric_train_test(overweight_long, "B")
pC <- plot_metric_train_test(underweight_long, "C")

# save plots
ggsave("J:/Research/Research/WorldBMI/ABSDATA/Performances/A_Obesity_TrainTest.png", pA, width=10, height=5, dpi=600)
ggsave("J:/Research/Research/WorldBMI/ABSDATA/Performances/B_Overweight_TrainTest.png", pB, width=10, height=5, dpi=600)
ggsave("J:/Research/Research/WorldBMI/ABSDATA/Performances/C_Underweight_TrainTest.png", pC, width=10, height=5, dpi=600)








library(readxl)
library(dplyr)

# define file path
file_path <- "J:/Research/Research/WorldBMI/ABSDATA/XGeoAI/IG_GeoAI_Results.xlsx"

# read the sheets
global_ig <- read_excel(file_path, sheet = "Global_IG")
regional_ig <- read_excel(file_path, sheet = "Regional_IG")

# check the first few rows
head(global_ig)
head(regional_ig)








library(readxl)
library(ggplot2)
library(dplyr)
library(tidyr)
library(patchwork)

# ===============================
# read Excel file
# ===============================
file_path <- "J:/Research/Research/WorldBMI/ABSDATA/XGeoAI/IG_GeoAI_Results.xlsx"
global_ig <- read_excel(file_path, sheet = "Global_IG")
regional_ig <- read_excel(file_path, sheet = "Regional_IG")
Country_IG = read_excel(file_path, sheet = "Country_IG")
# ===============================
# select top 5 features globally
# ===============================
top5_features <- global_ig %>% 
  arrange(desc(Mean_IG)) %>% 
  slice(1:5) %>% 
  pull(Feature)

# filter regional IG for top 5 features only
regional_top5 <- regional_ig %>%
  select(Region, all_of(top5_features)) %>%
  pivot_longer(-Region, names_to="Feature", values_to="IG_Value")

# ===============================
# color palettes
# ===============================
bar_colors <- c("darkred","navy","skyblue","orange","pink")  # professional distinct colors
heatmap_colors <- c("#7B9CCB","#4B6FA5","#2A4880","#0F2455", "navy")  # low to high IG

# ===============================
# horizontal bar for global IG top 5
# ===============================
global_top5 <- global_ig %>% filter(Feature %in% top5_features)
p_bar <- ggplot(global_top5, aes(x=reorder(Feature, Mean_IG), y=Mean_IG, fill=Feature)) +
  geom_bar(stat="identity", color="black") +
  geom_errorbar(aes(ymin=CI_Lower, ymax=CI_Upper), width=0.2) +
  coord_flip() +
  scale_fill_manual(values=bar_colors) +
  labs(x="Feature", y="Mean IG (95% CI)", title="A") +
  theme_minimal() +
  theme(legend.position="none",
        plot.title=element_text(size=16, face="bold"))

# ===============================
# heatmap plot (Features x Regions)
# ===============================
p_heatmap <- ggplot(regional_top5, aes(x=Region, y=factor(Feature, levels=rev(top5_features)), fill=IG_Value)) +
  geom_tile(color="black") +
  geom_text(aes(label=round(IG_Value, 3)), color="white", size=3.5) +
  scale_fill_gradientn(colors=heatmap_colors, name="IG Value") +
  labs(x="Region", y="Feature", title="B") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle=30, hjust=1),
        plot.title=element_text(size=16, face="bold"))

# ===============================
# combine bar and heatmap side by side
# ===============================
combined_plot <- p_bar + p_heatmap + plot_layout(ncol=2)

# display
combined_plot

# ===============================
# save combined plot
# ===============================
ggsave("J:/Research/Research/WorldBMI/ABSDATA/XGeoAI/Global_Regional_IG_Heatmap_Bar_Colors.png",
       combined_plot, width=8, height=4, dpi=600)









library(readxl)
library(dplyr)
library(sf)
library(ggplot2)
library(rnaturalearth)
library(rnaturalearthdata)

# ===============================
# read Excel file
# ===============================
file_path <- "J:/Research/Research/WorldBMI/ABSDATA/XGeoAI/IG_GeoAI_Results.xlsx"
Country_IG <- read_excel(file_path, sheet = "Country_IG")
global_ig <- read_excel(file_path, sheet = "Global_IG")

# ===============================
# standardize country names
# ===============================
Country_IG <- Country_IG %>%
  mutate(Country = case_when(
    Country == "United States" ~ "United States of America",
    Country == "Russia" ~ "Russian Federation",
    Country == "South Korea" ~ "Republic of Korea",
    Country == "North Korea" ~ "Dem. Rep. Korea",
    Country == "Czech Republic" ~ "Czechia",
    Country == "Iran" ~ "Iran (Islamic Republic of)",
    TRUE ~ Country
  ))

# ===============================
# select top 5 features globally
# ===============================
top5_features <- global_ig %>%
  arrange(desc(Mean_IG)) %>%
  slice(1:5) %>%
  pull(Feature)

# ===============================
# get world map (skip Antarctica)
# ===============================
world <- ne_countries(scale="medium", returnclass="sf") %>%
  filter(name != "Antarctica")

# ===============================
# create and save maps separately
# ===============================
for (feat in top5_features){
  
  map_data <- Country_IG %>%
    select(Country, all_of(feat)) %>%
    rename(IG_Value = all_of(feat))
  
  world_map <- left_join(world, map_data, by=c("name"="Country"))
  
  p <- ggplot(world_map) +
    geom_sf(aes(fill=IG_Value), color="black", size=0.1) +
    scale_fill_gradientn(colors=c("#D9E2F3","#7B9CCB","#4B6FA5","#2A4880","#0F2455"),
                         na.value="gray90",
                         name=paste0(feat," IG")) +
    theme_minimal() +
    labs(title=paste0("IG: ", feat)) +
    theme(axis.text=element_blank(),
          axis.ticks=element_blank(),
          panel.grid=element_blank(),
          plot.title=element_text(size=14, face="bold"),
          legend.position="bottom")
  
  # save each map individually
  ggsave(paste0("J:/Research/Research/WorldBMI/ABSDATA/XGeoAI/Country_IG_", feat, ".png"),
         p, width=10, height=6, dpi=600)
}
















library(readxl)
library(ggplot2)
library(dplyr)
library(tidyr)
library(patchwork)

# ===============================
# read Excel file
# ===============================
file_path <- "J:/Research/Research/WorldBMI/ABSDATA/XGeoAI/overweight_IG_GeoAI_Results.xlsx"
global_ig <- read_excel(file_path, sheet = "Global_IG")
regional_ig <- read_excel(file_path, sheet = "Regional_IG")
Country_IG = read_excel(file_path, sheet = "Country_IG")
# ===============================
# select top 5 features globally
# ===============================
top5_features <- global_ig %>% 
  arrange(desc(Mean_IG)) %>% 
  slice(1:5) %>% 
  pull(Feature)

# filter regional IG for top 5 features only
regional_top5 <- regional_ig %>%
  select(Region, all_of(top5_features)) %>%
  pivot_longer(-Region, names_to="Feature", values_to="IG_Value")

# ===============================
# color palettes
# ===============================
bar_colors <- c("darkred","navy","skyblue","orange","pink")  # professional distinct colors
heatmap_colors <- c("#7B9CCB","#4B6FA5","#2A4880","#0F2455", "navy")  # low to high IG

# ===============================
# horizontal bar for global IG top 5
# ===============================
global_top5 <- global_ig %>% filter(Feature %in% top5_features)
p_bar <- ggplot(global_top5, aes(x=reorder(Feature, Mean_IG), y=Mean_IG, fill=Feature)) +
  geom_bar(stat="identity", color="black") +
  geom_errorbar(aes(ymin=CI_Lower, ymax=CI_Upper), width=0.2) +
  coord_flip() +
  scale_fill_manual(values=bar_colors) +
  labs(x="Feature", y="Mean IG (95% CI)", title="C") +
  theme_minimal() +
  theme(legend.position="none",
        plot.title=element_text(size=16, face="bold"))

# ===============================
# heatmap plot (Features x Regions)
# ===============================
p_heatmap <- ggplot(regional_top5, aes(x=Region, y=factor(Feature, levels=rev(top5_features)), fill=IG_Value)) +
  geom_tile(color="black") +
  geom_text(aes(label=round(IG_Value, 3)), color="white", size=3.5) +
  scale_fill_gradientn(colors=heatmap_colors, name="IG Value") +
  labs(x="Region", y="Feature", title="D") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle=30, hjust=1),
        plot.title=element_text(size=16, face="bold"))

# ===============================
# combine bar and heatmap side by side
# ===============================
combined_plot <- p_bar + p_heatmap + plot_layout(ncol=2)

# display
combined_plot

# ===============================
# save combined plot
# ===============================
ggsave("J:/Research/Research/WorldBMI/ABSDATA/XGeoAI/overweightGlobal_Regional_IG_Heatmap_Bar_Colors.png",
       combined_plot, width=8, height=4, dpi=600)









library(readxl)
library(dplyr)
library(sf)
library(ggplot2)
library(rnaturalearth)
library(rnaturalearthdata)

# ===============================
# read Excel file
# ===============================
file_path <- "J:/Research/Research/WorldBMI/ABSDATA/XGeoAI/overweight_IG_GeoAI_Results.xlsx"
Country_IG <- read_excel(file_path, sheet = "Country_IG")
global_ig <- read_excel(file_path, sheet = "Global_IG")

# ===============================
# standardize country names
# ===============================
Country_IG <- Country_IG %>%
  mutate(Country = case_when(
    Country == "United States" ~ "United States of America",
    Country == "Russia" ~ "Russian Federation",
    Country == "South Korea" ~ "Republic of Korea",
    Country == "North Korea" ~ "Dem. Rep. Korea",
    Country == "Czech Republic" ~ "Czechia",
    Country == "Iran" ~ "Iran (Islamic Republic of)",
    TRUE ~ Country
  ))

# ===============================
# select top 5 features globally
# ===============================
top5_features <- global_ig %>%
  arrange(desc(Mean_IG)) %>%
  slice(1:5) %>%
  pull(Feature)

# ===============================
# get world map (skip Antarctica)
# ===============================
world <- ne_countries(scale="medium", returnclass="sf") %>%
  filter(name != "Antarctica")

# ===============================
# create and save maps separately
# ===============================
for (feat in top5_features){
  
  map_data <- Country_IG %>%
    select(Country, all_of(feat)) %>%
    rename(IG_Value = all_of(feat))
  
  world_map <- left_join(world, map_data, by=c("name"="Country"))
  
  p <- ggplot(world_map) +
    geom_sf(aes(fill=IG_Value), color="black", size=0.1) +
    scale_fill_gradientn(colors=c("#D9E2F3","#7B9CCB","#4B6FA5","#2A4880","#0F2455"),
                         na.value="gray90",
                         name=paste0(feat," IG")) +
    theme_minimal() +
    labs(title=paste0("IG: ", feat)) +
    theme(axis.text=element_blank(),
          axis.ticks=element_blank(),
          panel.grid=element_blank(),
          plot.title=element_text(size=14, face="bold"),
          legend.position="bottom")
  
  # save each map individually
  ggsave(paste0("J:/Research/Research/WorldBMI/ABSDATA/XGeoAI/overweightCountry_IG_", feat, ".png"),
         p, width=10, height=6, dpi=600)
}










library(readxl)
library(ggplot2)
library(dplyr)
library(tidyr)
library(patchwork)

# ===============================
# read Excel file
# ===============================
file_path <- "J:/Research/Research/WorldBMI/ABSDATA/XGeoAI/underweight_IG_GeoAI_Results.xlsx"
global_ig <- read_excel(file_path, sheet = "Global_IG")
regional_ig <- read_excel(file_path, sheet = "Regional_IG")
Country_IG = read_excel(file_path, sheet = "Country_IG")
# ===============================
# select top 5 features globally
# ===============================
top5_features <- global_ig %>% 
  arrange(desc(Mean_IG)) %>% 
  slice(1:5) %>% 
  pull(Feature)

# filter regional IG for top 5 features only
regional_top5 <- regional_ig %>%
  select(Region, all_of(top5_features)) %>%
  pivot_longer(-Region, names_to="Feature", values_to="IG_Value")

# ===============================
# color palettes
# ===============================
bar_colors <- c("darkred","navy","skyblue","orange","pink")  # professional distinct colors
heatmap_colors <- c("#7B9CCB","#4B6FA5","#2A4880","#0F2455", "navy")  # low to high IG

# ===============================
# horizontal bar for global IG top 5
# ===============================
global_top5 <- global_ig %>% filter(Feature %in% top5_features)
p_bar <- ggplot(global_top5, aes(x=reorder(Feature, Mean_IG), y=Mean_IG, fill=Feature)) +
  geom_bar(stat="identity", color="black") +
  geom_errorbar(aes(ymin=CI_Lower, ymax=CI_Upper), width=0.2) +
  coord_flip() +
  scale_fill_manual(values=bar_colors) +
  labs(x="Feature", y="Mean IG (95% CI)", title="E") +
  theme_minimal() +
  theme(legend.position="none",
        plot.title=element_text(size=16, face="bold"))

# ===============================
# heatmap plot (Features x Regions)
# ===============================
p_heatmap <- ggplot(regional_top5, aes(x=Region, y=factor(Feature, levels=rev(top5_features)), fill=IG_Value)) +
  geom_tile(color="black") +
  geom_text(aes(label=round(IG_Value, 3)), color="white", size=3.5) +
  scale_fill_gradientn(colors=heatmap_colors, name="IG Value") +
  labs(x="Region", y="Feature", title="F") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle=30, hjust=1),
        plot.title=element_text(size=16, face="bold"))

# ===============================
# combine bar and heatmap side by side
# ===============================
combined_plot <- p_bar + p_heatmap + plot_layout(ncol=2)

# display
combined_plot

# ===============================
# save combined plot
# ===============================
ggsave("J:/Research/Research/WorldBMI/ABSDATA/XGeoAI/underweightGlobal_Regional_IG_Heatmap_Bar_Colors.png",
       combined_plot, width=8, height=4, dpi=600)









library(readxl)
library(dplyr)
library(sf)
library(ggplot2)
library(rnaturalearth)
library(rnaturalearthdata)

# ===============================
# read Excel file
# ===============================
file_path <- "J:/Research/Research/WorldBMI/ABSDATA/XGeoAI/overweight_IG_GeoAI_Results.xlsx"
Country_IG <- read_excel(file_path, sheet = "Country_IG")
global_ig <- read_excel(file_path, sheet = "Global_IG")

# ===============================
# standardize country names
# ===============================
Country_IG <- Country_IG %>%
  mutate(Country = case_when(
    Country == "United States" ~ "United States of America",
    Country == "Russia" ~ "Russian Federation",
    Country == "South Korea" ~ "Republic of Korea",
    Country == "North Korea" ~ "Dem. Rep. Korea",
    Country == "Czech Republic" ~ "Czechia",
    Country == "Iran" ~ "Iran (Islamic Republic of)",
    TRUE ~ Country
  ))

# ===============================
# select top 5 features globally
# ===============================
top5_features <- global_ig %>%
  arrange(desc(Mean_IG)) %>%
  slice(1:5) %>%
  pull(Feature)

# ===============================
# get world map (skip Antarctica)
# ===============================
world <- ne_countries(scale="medium", returnclass="sf") %>%
  filter(name != "Antarctica")

# ===============================
# create and save maps separately
# ===============================
for (feat in top5_features){
  
  map_data <- Country_IG %>%
    select(Country, all_of(feat)) %>%
    rename(IG_Value = all_of(feat))
  
  world_map <- left_join(world, map_data, by=c("name"="Country"))
  
  p <- ggplot(world_map) +
    geom_sf(aes(fill=IG_Value), color="black", size=0.1) +
    scale_fill_gradientn(colors=c("#D9E2F3","#7B9CCB","#4B6FA5","#2A4880","#0F2455"),
                         na.value="gray90",
                         name=paste0(feat," IG")) +
    theme_minimal() +
    labs(title=paste0("IG: ", feat)) +
    theme(axis.text=element_blank(),
          axis.ticks=element_blank(),
          panel.grid=element_blank(),
          plot.title=element_text(size=14, face="bold"),
          legend.position="bottom")
  
  # save each map individually
  ggsave(paste0("J:/Research/Research/WorldBMI/ABSDATA/XGeoAI/underweightCountry_IG_", feat, ".png"),
         p, width=10, height=6, dpi=600)
}


