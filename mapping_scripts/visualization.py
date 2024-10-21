import pandas as pd
df = pd.read_csv('/Users/chloecrozier/Desktop/cevac_facillities/data/workorder.csv', low_memory=False)

columns = df.keys()
# print(columns)

selected_columns = df[['WO_WORKORDER', 'WO_BARCODE', 'WO_STATUS', 'WO_ENTRY_DATE', 'WO_DESC', 'WO_TYPE_WITH_DESCRIPTION', 'WO_CATEGORY_WITH_DESC', 'WO_CREATED_BY', 'WO_ORGANIZATION_WITH_DESC', 'WO_REQUESTOR_WITH_DESC', 'WO_CONTACT_PHONE', 'WO_CONTACT_EMAIL', 'WO_REGION_DESC', 'WO_FACILITY', 'WO_BUILDING', 'WO_BUILDING_DESC', 'WO_SHOP', 'WO_SHOP_DESC', 'WO_SHOP_PERSON_WITH_DESC', 'WO_STATUS_DATE']]
# print(selected_columns)
open_rows = selected_columns[selected_columns['WO_STATUS'] == 'Open']
print(open_rows)

