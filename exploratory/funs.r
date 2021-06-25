#labels
my_labs <- theme(
  plot.title = element_text(size = 12),
  plot.subtitle = element_text(size=10),
  plot.caption = element_text(face = "italic"))

#legend remover
g_legend <- function(a.gplot){
  tmp <- ggplot_gtable(ggplot_build(a.gplot))
  leg <- which(sapply(tmp$grobs, function(x) tmp$name) == "guide-box")
  legend <- tmp$grobs[[leg]]
  return(legend)}

# industry2 to name vector conversion:
mapping <- function(industry2){
  dict <-c(
  "11" = "Agriculture",
  "21" = "Mining",
  "22" = 	"Utilities",
  "23" =	"Construction",
  "31" = "Manufacturing", "32" = "Manufacturing" , "33" = "Manufacturing",
  "42" = "Trade",
  "44" = "Retail",
  "45" = "Retail",
  "48" = "Transportation",
  "49" = "Transportation",
  "51" = "Information",
  "52" = "Finance",
  "53" = "Real Estate",
  "54" = "Professional",
  "55" = "Management",
  "56" = "Administrative",
  "61" = "Educational",
  "62" = "Healthcare",
  "71" = "Entertainment",
  "72" = "Accommodation",
  "81" = "Other",
  "92" = "Public"
  )
  output <- dict[as.character(industry2)]
  names(output) <- NULL
  return  (output)
}

# COMPACT industry2 to name vector conversion:
mapping_compact <- function(industry2){
  dict <-c(
  "11" = "Other",
  "21" = "Other",
  "22" = 	"Other",
  "23" =	"Other",
  "31" = "Other", "32" = "Other" , "33" = "Other",
  "42" = "Other",
  "44" = "Retail",
  "45" = "Retail",
  "48" = "Other",
  "49" = "Other",
  "51" = "Information",
  "52" = "Finance",
  "53" = "Other",
  "54" = "Other",
  "55" = "Other",
  "56" = "Other",
  "61" = "Educational",
  "62" = "Healthcare",
  "71" = "Other",
  "72" = "Other",
  "81" = "Other",
  "92" = "Public"
  )
  output <- dict[as.character(industry2)]
  names(output) <- NULL
  return  (output)
}

# NEW industry name to reduced name vector conversion:
mapping_new <- function(industry.name){
  dict <-c(
  "Retail " = "Retail",
  "Information " = "Information",
  "Finance " = "Finance",
  "Educational " = "Educational",
  "Healthcare " = "Healthcare",
  "Public " = "Public",
  "Agriculture " = "Other",
  "Mining " = "Other",
  "Utilities " = "Other",
  "Accomodation " = "Other",
  "Entertainment " = "Other",
  "Professional " = "Other",
  "Real Estate " = "Other",
  "Administrative " = "Other",
  "Management " = "Other",
  "Construction " = "Other",
  "Manufacturing " = "Other",
  "Transportation " = "Other",
  "Trade " = "Other",
  "Unknown" = "Unknown"
  )
  output <- dict[industry.name]
  names(output) <- NULL
  return  (output)
}
