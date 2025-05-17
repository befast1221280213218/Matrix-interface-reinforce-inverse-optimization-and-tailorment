library(linkET)
library(dplyr)
library(ggplot2)
library(vegan)
library(readxl)
# 
install.packages("RColorBrewer")


library(RColorBrewer)


data <- read_excel("****.xlsx")


data_num <- data %>% select(where(is.numeric))

data_dist <- dist(data_num, method = "euclidean")


mantel <- mantel_test(data_num, data_num,  
                      spec_select = list(
                        E = A  
                      )) %>%
  mutate(
    rd = cut(r, breaks = c(-Inf, 0.2, 0.4, Inf), labels = c("< 0.2", "0.2 - 0.4", ">= 0.4")),
    pd = cut(p, breaks = c(-Inf, 0.01, 0.05, Inf), labels = c("< 0.01", "0.01 - 0.05", ">= 0.05"))
  )


red_blue <- brewer.pal(11, "RdBu")


p <- qcorrplot(correlate(data_num), type = "lower", diag = FALSE) +      
  geom_square() +
  geom_couple(aes(colour = pd, size = rd),  
              data = mantel,  
              curvature = nice_curvature()) +
  scale_fill_gradientn(colours = rev(c("#FEB3AE", "#A2C6F1")))+
  scale_size_manual(values = c(0.5, 1.5, 3)) +
  scale_colour_manual(values = c( "#FEB3AE","#A2C6F1", "#D3D3D3")) +
  guides(size = guide_legend(title = "Mantel's r",  
                             override.aes = list(colour = "grey35"),  
                             order = 2),  
         colour = guide_legend(title = "Mantel's p",  
                               override.aes = list(size = 3),  
                               order = 1),  
         fill = guide_colorbar(title = "Pearson's r", order = 3)) +
  theme(
    text = element_text(size = 10, family = "Arial"),
    plot.title = element_text(size = 10, colour = "black", family = "Arial", hjust = 0.5),
    legend.title = element_text(color = "black", family = "Arial", size = 10),
    legend.text = element_text(color = "black", family = "Arial", size = 10),
    axis.text.y = element_blank(),
    axis.text.x = element_text(size = 16, color = "black", family = "Arial", vjust = 0.5, hjust = 0.5, angle = 0),
    legend.position = "left"  
  )


print(p)
