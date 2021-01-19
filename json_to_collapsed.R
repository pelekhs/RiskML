setwd("C:/Ubuntu/Dev/Windows/VERIS/vcdb_datamining")
library(verisr)
library(verisr2)

# jsontoveris
vcdb.dir <- "../VCDB/data/json/validated/"
# may optionally load a custom json schema file.
if (interactive()) { # show progress bar if the session is interactive
  vcdb <- json2veris(dir=vcdb.dir, schema="../VCDB/vcdb-merged.json", progressbar=TRUE)
} else {
  vcdb <- json2veris(dir=vcdb.dir, schema="../VCDB/vcdb-merged.json")  
}



# Collapse vcdb to a more compact data.frame (df)
df <- collapse_vcdb(vcdb)

# Column pruning

## Actor


df$actor.partner.name <- NULL
df$actor.partner.industry <- NULL
#df$actor.partner.motive <- NULL
df$actor.partner.notes <- NULL
df$actor.partner.region <- NULL
df$actor.partner.country <- df$actor.partner.country
df$actor.partner.industry2 <- NULL
df$actor.partner.industry3 <- NULL

df$actor.external.industry <- NULL
#df$actor.external.motive <- NULL
df$actor.external.name <- NULL
df$actor.external.notes <- NULL
df$actor.external.country <- df$actor.external.country
df$actor.external.region <- NULL
df$actor.external.variety <- df$actor.external.variety 

df$actor.internal.job_change <- df$actor.internal.job_change
df$actor.internal.notes <- NULL
df$actor.internal.variety <- df$actor.internal.variety
#df$actor.internal.motive <- NULL

df$actor.unknown.notes <- NULL

## Attribute

# render attributes one hot so as to avoid "Multiple" class
df$attribute <- NULL
df$attribute.confidentiality <- vcdb$attribute.Confidentiality
df$attribute.integrity <- vcdb$attribute.Integrity
df$attribute.availability <- vcdb$attribute.Availability
df$attribute.availability.variety <-NULL
df$attribute.confidentiality.primary_attribute <- NULL
df$attribute.integrity.notes <- NULL
df$attribute.unknown.notes <- NULL
df$attribute.availability.notes <- NULL
df$attribute.confidentiality.notes <- NULL
# most are unknown
df$attribute.integrity.variety <- NULL
# most are unknown
df$plus.attribute.confidentiality.data_abuse <- NULL
df$attribute.unknown.result <- NULL
df$plus.attribute.confidentiality.credit_monitoring <- NULL


## Victim

df$victim.state <- NULL
df$victim.industry2 <- NULL
df$victim.industry3 <- NULL
df$victim.locations_affected <- NULL
# keep it as impact metric
df$victim.secondary.victim_id <- df$victim.secondary.victim_id
df$victim.secondary.notes <- NULL
df$victim.notes <- NULL
df$victim.region <- NULL
df$victim.secondary.victim_id <- NULL


## Assets


# keep them for future reports and stats (no predictive power, too many uknown and NAN)
df$asset.cloud <- df$asset.cloud
df$asset_os <- df$asset_os
# keep it in in case you can infer the amount from other info (most are 0)
df$asset.total_amount <- df$asset.total_amount
# garbage
df$asset.hosting <- df$asset.hosting
df$asset.country <- NULL
df$asset.management <- NULL
df$asset.notes <- NULL
df$asset.ownership <- NULL
df$asset.primary_asset <- NULL
df$asset.role <- NULL
df$plus.asset.total <- NULL


# Action


df$action.error.notes <- NULL
df$action.hacking.notes <- NULL
df$action.malware.notes <- NULL
df$action.physical.notes <- NULL
df$action.environmental.notes <- NULL
df$action.environmental.variety <- NULL
df$action.unknown.notes <- NULL
df$action.social.notes <- NULL
df$action.misuse.notes <- NULL
df$cost_corrective_action <- NULL
df$plus.asset.total <- NULL
# keep it in order to set risk levels according to users/targets that 
# have access rights to an asset
df$action.social.target <- df$action.social.target

# Plus


# keep it for filtering purposes
df$plus.analysis_status <- df$plus.analysis_status

df$plus.analyst_notes <- NULL
df$plus.analyst <- NULL
# refers to VCDB issues -> irrelevant
df$plus.github <- NULL
df$plus.attack_difficulty_initial <- NULL
df$plus.attack_difficulty_subsequent <- NULL
df$plus.attack_difficulty_legacy <- NULL
df$plus.timeline.notification.month <- NULL
df$plus.timeline.notification.year <- NULL
df$plus.timeline.notification.day <- NULL
df$plus.unknown_unknowns <- NULL
df$plus.pci.req_1 <- NULL
df$plus.pci.req_2 <- NULL
df$plus.pci.req_3 <- NULL
df$plus.pci.req_4 <- NULL
df$plus.pci.req_5 <- NULL
df$plus.pci.req_6 <- NULL
df$plus.pci.req_7 <- NULL
df$plus.pci.req_8 <- NULL
df$plus.pci.req_9 <- NULL
df$plus.pci.req_10 <- NULL
df$plus.pci.req_11 <- NULL
df$plus.pci.req_12 <- NULL
df$plus.unfollowed_policies <- NULL
df$plus.sub_source <- NULL
df$plus.security_maturity <- NULL
df$plus.pci.merchant_level <- NULL
df$plus.master_id <- NULL
df$plus.investigator <- NULL
df$plus.created <- NULL
df$plus.modified <- NULL
df$plus.plus.dbir_year <- NULL
df$plus.pci.compliance_status <- NULL
df$plus.dbir_year <- NULL
df$plus.analysis_status <- NULL

## Impact


# Catastrophic      Damaging   Distracting Insignificant       Painful       Unknown 
#            2            30            42           121            20          8474 
df$impact.overall_rating <- NULL
df$impact.notes <- NULL
# all unknown
df$impact.loss.rating <- NULL
# all unknown
df$impact.loss.variety <- NULL



## Timeline


df$timeline.incident.time <- NULL
df$timeline.incident.day <- NULL
df$timeline.exfiltration.unit <- NULL
df$timeline.exfiltration.value <- NULL

# How long from the first action to the first compromise of an attribute?
df$timeline.compromise.unit <- df$timeline.compromise.unit
df$timeline.compromise.value <- df$timeline.compromise.value
# How long from compromise until the incident was discovered by the victim organization?
df$timeline.discovery.unit <- df$timeline.discovery.unit
df$timeline.discovery.value <- df$timeline.discovery.value


## Value Chain:


df["value_chain.cash-out.notes"] <- NULL
df["value_chain.non-distribution services.notes"] <- NULL
df["value_chain.money laundering.variety"]<- NULL
df$value_chain.distribution.notes <- NULL
df$value_chain.distribution.notes <- NULL
df$value_chain.development.notes <- NULL
df$value_chain.targeting.notes <- NULL
df$value_chain.targeting.variety <- NULL
df$value_chain.development.variety <- NULL
df$value_chain.distribution.variety <- NULL



## Discovery_method and notes

df$discovery_method <- NULL
df$discovery_method.external.variety <- NULL
df$discovery_method.internal.variety <- NULL
df$discovery_method.partner.variety <- NULL
df$discovery_notes <- NULL



## Misc

df$control_failure <- NULL
df$confidence <- NULL
df$notes <- NULL
df$schema_version <- NULL
df$summary <- NULL
df$security_incident <- NULL
df$source_id <- NULL
df$targeted <- NULL
df["distribution services.variety"] <- NULL
df$out.variety <-NULL


## Remove 7 environmental incidents

#remove environmental
#df <- df[df$action != "Environmental",]



## Save csv for python processing

con <- file("C:/Ubuntu/Dev/Windows/VERIS/csv/Rcollapsed.csv", encoding="UTF-8")
write.csv(df, file=con, row.names = T)