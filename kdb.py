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

### Original
query_to_get_event = """
({string connDetails}) (({dtRange};{eventSym})
 select from latestEconomicEvent
 where sym = eventSym,
       date = .z.D - 1,
       eventDate.date within dtRange
);dtRange;eventSym)
"""
result = to_pd(kx.q(
    query_to_get_event,
    static_data_conn_details,
    event_date_range,
    event
))

# new 1
query_to_get_event = f"""
{{{static_data_conn_details}}} (({{{event_date_range};`{event}}})
 select from latestEconomicEvent
 where sym = `{event},
       date = .z.D - 1,
       eventDate.date within {event_date_range}
);{event_date_range};`{event})
"""
result = to_pd(kx.q(query_to_get_event))


# new2
query_to_get_event = """
({connDetails}) (({dtRange};`$eventSym)
 select from latestEconomicEvent
 where sym = `$eventSym,
       date = .z.D - 1,
       eventDate.date within dtRange
);dtRange;`$eventSym)
"""

result = to_pd(kx.q(
    query_to_get_event,
    static_data_conn_details,
    event_date_range,
    event
))


