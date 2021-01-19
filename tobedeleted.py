# # Initial drops
# collapsed.drop(columns="victim.country", inplace=True)
# collapsed.drop(columns="victim.victim_id", inplace=True)
# collapsed.drop(columns="action.error.vector", inplace=True)
# collapsed.drop(columns="action.error.variety", inplace=True)
# collapsed.drop(columns="action.hacking.cve", inplace=True)
# collapsed.drop(columns="action.malware.cve", inplace=True)
# collapsed.drop(columns="action.malware.name", inplace=True)
#
# #result
# collapsed.drop(columns="action.hacking.result", inplace=True)
# collapsed.drop(columns="action.malware.result", inplace=True)
# collapsed.drop(columns="action.physical.result", inplace=True)
# collapsed.drop(columns="action.misuse.result", inplace=True)
# collapsed.drop(columns="action.social.result", inplace=True)
# collapsed.drop(columns="action.unknown.result", inplace=True)
# collapsed.drop(columns="plus.attribute.confidentiality.partner_data", inplace=True)
# collapsed.drop(columns="reference", inplace=True)
# collapsed.drop(columns="plus.asset.os", inplace=True)
#
# #timeline
# collapsed.drop(columns="timeline.incident.year", inplace=True)
# collapsed.drop(columns="timeline.incident.month", inplace=True)
#
# collapsed.drop(columns="asset.cloud", inplace=True)
