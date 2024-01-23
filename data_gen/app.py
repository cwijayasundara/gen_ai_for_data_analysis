import pandas as pd
import plotly.express as px
import plotly.io as pio
import random

pio.templates.default = 'simple_white'
pio.renderers.default = "png"  # to have static images
import datetime
import tqdm


# modelling retention
def get_retention(a, b, c, d, periods):
    return a + 1. / (b + c * periods ** d)


def get_retention_same_event(a, c, d, periods):
    b = 1. / (1 - a)
    return get_retention(a, b, c, d, periods)


sample_df = pd.DataFrame()
sample_df['periods'] = range(30)

sample_df = pd.DataFrame()
sample_df['periods'] = range(24)

product_coefs = [
    (0, 0.55, 2),
    (0.02, 0.3, 1.6),
    (0.04, 0.3, 1.5),
    (0.01, 1.3, 1)
]

for i in range(len(product_coefs)):
    sample_df['product' + str(i + 1)] = sample_df.periods.map(
        lambda x: get_retention_same_event(product_coefs[i][0], product_coefs[i][1],
                                           product_coefs[i][2], x)
    )

sample_df = sample_df.set_index('periods')
px.line(sample_df.map(lambda x: None if x < 0.01 else 100 * x).loc[0:],
        title='Monthly retention',
        labels={'value': 'retention, %',
                'periods': '# month',
                'variable': 'product'})

# Modelling new users

weekly_coefs = {
    0: 1.0,
    1: 0.9942430174015676,
    2: 0.9820212874783774,
    3: 0.9790313740157027,
    4: 0.9385562774857475,
    5: 0.7855713201801697,
    6: 0.8163537550287501
}

new_users_df1 = pd.DataFrame()
new_users_df1['date'] = pd.date_range('2021-01-01', '2023-12-31')
new_users_df1['x'] = range(new_users_df1.shape[0])

new_users_df1['trend'] = new_users_df1.x.map(
    lambda x: 1 / (0.0036 + (x + 1) ** -1.3)
)
new_users_df1.drop('x', axis=1, inplace=True)


def get_new_users(date, trend):
    return int((weekly_coefs[date.weekday()] + 0.1 * random.random()) * trend)


new_users_df1['new_users'] = list(map(
    get_new_users,
    new_users_df1.date,
    new_users_df1.trend
))

px.line(new_users_df1.set_index('date'))

new_users_df2 = pd.DataFrame()
new_users_df2['date'] = pd.date_range('2023-02-14', '2023-12-31')
new_users_df2['x'] = range(new_users_df2.shape[0])

new_users_df2['trend'] = new_users_df2.x.map(
    lambda x: 1 / (0.0003 + (x + 1) ** -1.25)
)
new_users_df2.drop('x', axis=1, inplace=True)

new_users_df2['new_users'] = list(map(
    get_new_users,
    new_users_df2.date,
    new_users_df2.trend
))

px.line(new_users_df2.set_index('date'))

new_users_df3 = pd.DataFrame()
new_users_df3['date'] = pd.date_range('2022-02-24', '2023-12-31')
new_users_df3['x'] = range(new_users_df3.shape[0])

new_users_df3['trend'] = new_users_df3.x.map(
    lambda x: 1 / (0.0023 + (x + 1) ** -1.8)
)
new_users_df3.drop('x', axis=1, inplace=True)

new_users_df3['new_users'] = list(map(
    get_new_users,
    new_users_df3.date,
    new_users_df3.trend
))

px.line(new_users_df3.set_index('date'))

# Modelling data

users_lst1 = []
last_id = 1

for rec in new_users_df1.to_dict('records'):
    for _ in range(rec['new_users']):
        users_lst1.append(
            {
                'user_id': last_id,
                'cohort': rec['date']
            }
        )
        last_id += 1

user_activity1 = []
for rec in tqdm.tqdm(users_lst1):
    user_id = rec['user_id']
    cohort = rec['cohort']
    for date in pd.date_range(rec['cohort'], '2023-12-31'):
        num_day = (date - rec['cohort']).days
        if cohort < datetime.datetime(2022, 2, 24):
            params = (0.01, 0.3, 1.6)
        elif cohort < datetime.datetime(2022, 12, 18):
            params = (0.02, 0.3, 1.6)
        else:
            params = (0.03, 0.3, 1.4)
        if random.random() <= get_retention_same_event(params[0], params[1], params[2], num_day) \
                * weekly_coefs[date.weekday()] * (1 + (random.random() - 0.5) * 2 * 0.3):
            user_activity1.append(
                {
                    'user_id': user_id,
                    'date': date
                }
            )

act_df1 = pd.DataFrame(user_activity1)

print(act_df1.shape)

px.line(act_df1.groupby('date')[['user_id']].count())

print(act_df1.shape[0])

users_lst2 = []
last_id = 1

for rec in new_users_df2.to_dict('records'):
    for _ in range(rec['new_users']):
        users_lst2.append(
            {
                'user_id': last_id,
                'cohort': rec['date']
            }
        )
        last_id += 1

user_activity2 = []
for rec in tqdm.tqdm(users_lst2):
    user_id = rec['user_id']
    cohort = rec['cohort']
    for date in pd.date_range(rec['cohort'], '2023-12-31'):
        num_day = (date - rec['cohort']).days
        params = (0.0, 0.55, 1.1)
        if random.random() <= get_retention_same_event(params[0], params[1], params[2], num_day) \
                * weekly_coefs[date.weekday()] * (1 + (random.random() - 0.5) * 2 * 0.3):
            user_activity2.append(
                {
                    'user_id': user_id,
                    'date': date
                }
            )

act_df2 = pd.DataFrame(user_activity2)
print(act_df2.shape[0])

users_lst3 = []
last_id = 1

for rec in new_users_df3.to_dict('records'):
    for _ in range(rec['new_users']):
        users_lst3.append(
            {
                'user_id': last_id,
                'cohort': rec['date']
            }
        )
        last_id += 1

user_activity3 = []
for rec in tqdm.tqdm(users_lst3):
    user_id = rec['user_id']
    cohort = rec['cohort']
    for date in pd.date_range(rec['cohort'], '2023-12-31'):
        num_day = (date - rec['cohort']).days
        params = (0.01, 1.3, 1)
        if random.random() <= get_retention_same_event(params[0], params[1], params[2], num_day) \
                * weekly_coefs[date.weekday()] * (1 + (random.random() - 0.5) * 2 * 0.3):
            user_activity3.append(
                {
                    'user_id': user_id,
                    'date': date
                }
            )

act_df3 = pd.DataFrame(user_activity3)
print(act_df3.shape[0])

act_df1['user_id'] = act_df1['user_id'] + 10 ** 6
act_df2['user_id'] = act_df1['user_id'] + 10 ** 6 * 2
act_df3['user_id'] = act_df1['user_id'] + 10 ** 6 * 3

act_df1['os'] = 'Windows'
act_df2['os'] = 'iOS'
act_df3['os'] = 'Android'

print(act_df1.shape[0], act_df2.shape[0], act_df3.shape[0])

act_df = pd.concat(
    [act_df1, act_df2, act_df3]
)

act_df.to_csv('full_data.csv', sep='\t')

print(act_df.shape[0])

act_df = pd.read_csv('full_data.csv', sep='\t')
act_df['session_id'] = list(map(lambda x: x + 1, range(act_df.shape[0])))
act_df['browser'] = act_df.user_id.map(
    lambda x: random.choices(['Safari', 'Chrome', 'Firefox'], weights=[0.3, 0.6, 0.1], k=1)[0])
act_df['session_duration'] = act_df.user_id.map(lambda x: random.randrange(0, 1000))
act_df['is_fraud'] = act_df.user_id.map(lambda x: random.choices([0, 1], weights=[0.99, 0.01], k=1)[0])
act_df['revenue'] = act_df.user_id.map(lambda x: random.randrange(0, 100000) / 10)
act_df['revenue'] = act_df.revenue.map(lambda x: 0 if random.choices([0, 1], weights=[0.7, 0.3], k=1)[0] == 0 else x)
act_df.to_csv('full_data.csv', sep='\t', index=False)

print(act_df.head())

# Generating users table
users_df = act_df[['user_id']].drop_duplicates()
users_df['country'] = users_df.user_id.map(
    lambda x: random.choices(['United Kingdom', 'Germany', 'France', 'Netherlands'], k=1)[0])
users_df['active'] = act_df.user_id.map(lambda x: random.choices([1, 0], weights=[0.95, 0.05], k=1)[0])
users_df['age'] = users_df.user_id.map(lambda x: random.randrange(18, 90))
users_df.to_csv('full_users_data.csv', sep='\t', index=False)

print(users_df.head())

# Uploading data to ClickHouse
import requests
import io

CH_HOST = 'http://localhost:8123'
pd.set_option('display.max_colwidth', 1000)


def get_clickhouse_data(query, host=CH_HOST, connection_timeout=1500):
    r = requests.post(host, params={'query': query},
                      timeout=connection_timeout)

    if r.status_code == 200:
        return r.text
    else:
        raise ValueError(r.text)


def get_clickhouse_df(query, host=CH_HOST, connection_timeout=1500):
    data = get_clickhouse_data(query, host, connection_timeout)
    df = pd.read_csv(io.StringIO(data), sep='\t')
    return df


def upload_data_to_clickhouse(table, content, host=CH_HOST):
    content = content.encode('utf-8')
    query_dict = {
        'query': 'INSERT INTO ' + table + ' FORMAT TabSeparatedWithNames '
    }
    r = requests.post(host, data=content, params=query_dict)
    result = r.text
    if r.status_code == 200:
        return result
    else:
        raise ValueError(r.text)


print(get_clickhouse_data('select 1'))

q = '''
create database ecommerce
'''

print(get_clickhouse_data(q))

q = '''
drop table if exists ecommerce.users
'''

print(get_clickhouse_data(q))

q = '''
CREATE TABLE ecommerce.users
(
  `user_id` UInt64,
  `country` String,
  `is_active` UInt8,
  `age` UInt64
)
ENGINE = Log
SETTINGS index_granularity = 8192
'''

print(get_clickhouse_data(q))

upload_data_to_clickhouse('ecommerce.users',
                          users_df.rename(columns={'active': 'is_active'}).to_csv(index=False, sep='\t'))

q = '''
select 
    country, 
    count(1) as users
from ecommerce.users
group by country 
with totals
format TabSeparatedWithNames
'''

print(get_clickhouse_df(q))
print(act_df.head())

q = '''
CREATE TABLE ecommerce.sessions
(
  `user_id` UInt64,
  `session_id` UInt64,
  `action_date` Date,
  `session_duration` UInt64,
  `os` String,
  `browser` String,
  `is_fraud` UInt8,
  `revenue` Float32
)
ENGINE = MergeTree
PARTITION BY toYYYYMM(action_date)
ORDER BY (action_date, intHash32(user_id))
SAMPLE BY intHash32(user_id)
SETTINGS index_granularity = 8192
'''

print(get_clickhouse_data(q))

upload_data_to_clickhouse('ecommerce.sessions',
                          act_df.rename(columns={'date': 'action_date'}).to_csv(index=False, sep='\t'))

q = '''
select 
    os, 
    count(1) as users
from ecommerce.sessions
group by os 
with totals
format TabSeparatedWithNames
'''

print(get_clickhouse_df(q))

q = '''
select * 
from ecommerce.sessions
limit 5
format TabSeparatedWithNames
'''

print(get_clickhouse_df(q))

q = '''
select * 
from ecommerce.users
limit 5
format TabSeparatedWithNames
'''

print(get_clickhouse_df(q))
