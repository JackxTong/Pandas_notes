import numpy as np
from datetime import date

# Setup
database = 's1:29502'  # economic event DB
static_data_conn_details = f':kdb-rates-gem-{database}:produser:prodpass'
event = "US Core PCE yy (US)"

# Date filters
today = np.datetime64(date.today())
yesterday = today - np.timedelta64(1, 'D')
event_date_range = [np.datetime64("2024-12-16"), np.datetime64("2025-05-20")]


from datetime import date

# Values
event = "US Core PCE yy (US)"
start_date = "2024.12.16"
end_date = "2025.05.20"

# Format the q query directly
query_to_get_event = f'''
select from latestEconomicEvent 
where sym = `{"`" + event.replace(" ", "_").replace("(", "").replace(")", "")}`, 
      date = .z.D - 1, 
      eventDate.date within ({start_date}; {end_date})
'''

query_to_get_event = f'''
select from latestEconomicEvent 
where sym = $"US Core PCE yy (US)", 
      date = .z.D - 1, 
      eventDate.date within ({start_date}; {end_date})
'''
