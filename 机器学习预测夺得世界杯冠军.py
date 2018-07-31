
# coding: utf-8

# # 使用机器学习预测世界杯冠军 🏆


# ## 项目大纲

# - 获取本届世界杯小组赛和淘汰赛的比赛结果
# - 获取球队在本届世界杯的比赛统计数据
# - 获取球队在历届世界杯的比赛统计数据
# - 使用人工神经网络进行预测

# ## 数据来源

# - 搜狐体育 2018 俄罗斯世界杯实时 [统计数据](http://data.2018.sohu.com/)。
# - 外部数据可能会因为 API 变动而失效。

# --- 

# ## 获取本届世界杯小组赛和淘汰赛的比赛结果

# - 数据地址：http://data.2018.sohu.com/game-schedule.html?index=3

# ### 解析 JSON 数据

# In[1]:


import requests

# 解析 JSON 数据
play_raw = requests.get("http://api.data.2018.sohu.com/api/schedule/time")


# In[2]:


import json

play_json = json.loads(play_raw.text)


# In[3]:


play_json['result'][0]


# In[4]:


import pandas as pd

play_df = pd.read_json(json.dumps(play_json['result']))


# In[5]:


play_score = play_df[['home_team_name', 'visiting_team_name', 'home_team_score', 'visiting_team_score']].iloc[:-1]
play_score.tail()


# ### 根据比分情况，为每一场比赛添加标签

# In[6]:


play_score.loc[play_score['home_team_score'] > play_score['visiting_team_score'], 'results'] = '胜利'
play_score.loc[play_score['home_team_score'] == play_score['visiting_team_score'], 'results'] = '平局'
play_score.loc[play_score['home_team_score'] < play_score['visiting_team_score'], 'results'] = '失败'


# In[7]:


play_score.head()


# ## 获取球队在本届世界杯的比赛统计数据

# 获取球队在本届世界杯的比赛统计数据，这些数据包括赢球场次、输球场次、比赛次数、进球数量、失球数量等。这些指标用于反映球队的整体实力。
# 
# - 数据地址：http://data.2018.sohu.com/

# ### 得到各国家队整体输赢数据

# In[8]:


team_raw = requests.get("http://api.data.2018.sohu.com/api/scores/index")


# In[9]:


team_json = json.loads(team_raw.text)

team_json['result'][0]


# In[10]:


team_df = pd.read_json(json.dumps(team_json['result']))
team_df = team_df[['name_cn', 'wins', 'losses', 'ties', 'points', 'goals_for', 'goals_against']]
team_df_reindex = pd.DataFrame(team_df).set_index('name_cn')
team_df_reindex.head()


# ### 得到各国家队得分详细统计数据

# - 数据地址：http://data.2018.sohu.com/list.html?type=team&category=2

# In[11]:


goal_raw = requests.get("http://api.data.2018.sohu.com/api/rank/team?category_id=2")


# In[12]:


goal_json = json.loads(goal_raw.text)

goal_json['result']['list'][0]


# In[13]:


goal_df = pd.read_json(json.dumps(goal_json['result']['list']))
goal_df = goal_df[['name_cn', 'games_played', 'goals', 'opponent_goals', 'shots', 'shots_on_goal', 
                   'fouls', 'offsides', 'touches_passes', 'free_kicks', 'corner_kicks', 'duelstackletotal',
                   'yellow_cards', 'red_cards']]
goal_df_reindex = pd.DataFrame(goal_df).set_index('name_cn')
goal_df_reindex.head()


# ## 获取球队在历届世界杯的比赛统计数据

# - 数据地址：http://data.2018.sohu.com/team-map.html

# In[14]:


team_history_raw = requests.get("http://api.data.2018.sohu.com/api/team/list")


# In[15]:


team_history_json = json.loads(team_history_raw.text)

team_history_json['result'][1]


# ### 得到以国家名称为索引，各国家历年参加世界杯及名次情况

# In[16]:


team_history_df = pd.read_json(json.dumps(team_history_json['result']))
team_history_df = team_history_df[['name_cn', 'global_rank', 'crowns', 'presents']]
team_history_df_reindex = pd.DataFrame(team_history_df).set_index('name_cn')
team_history_df_reindex.head()


# ## 合并数据

# 合并 `team_df_reindex`，`goal_df_reindex`，`team_history_df_reindex`。

# In[17]:


team_merge = pd.concat([team_df_reindex, goal_df_reindex, team_history_df_reindex], axis=1)

team_merge.head()


# ## 将各国家赛况信息拼合到对阵双方比分结果数据集中

# In[18]:


home_team_df = team_merge.reindex(play_score['home_team_name'])
visiting_team_df = team_merge.reindex(play_score['visiting_team_name'])


# In[19]:


home_visiting_team_df = pd.concat([home_team_df.reset_index(), visiting_team_df.reset_index()], axis=1)
home_visiting_team_df.head()


# In[20]:


play_score_new = pd.concat([home_visiting_team_df, play_score.iloc[:, -1:]], axis=1).drop(['home_team_name', 'visiting_team_name'], axis=1)
play_score_new.head()


# ### 数据归一化处理

# Min-Max Normalization 对原始数据的线性变换，使结果值映射到 `0-1` 之间：

# $$\hat x=\frac{x-x_{min}}{x_{max}-x_{min}}$$

# In[21]:


play_score_temp = play_score_new.iloc[:, :-1]
play_score_normal = (play_score_temp - play_score_temp.min()) / (play_score_temp.max() - play_score_temp.min())
play_score_normal = pd.concat([play_score_normal, play_score_new.iloc[:, -1]], axis=1)
play_score_normal.head()


# ## 使用人工神经网络进行预测

# In[22]:


X = play_score_normal.iloc[:, :-1] # 特征
y = play_score_normal.iloc[:, -1] # 目标


# In[23]:


from sklearn.neural_network import MLPClassifier

# 定义人工神经网络分类器
model = MLPClassifier(max_iter=1000)


# In[24]:


from sklearn.model_selection import cross_val_score

# 交叉验证，评估模型可靠性
cvs = cross_val_score(model, X, y, cv=5)
cvs


# In[25]:


import numpy as np

# 求得交叉验证结果平均值
np.mean(cvs)


# In[26]:


model.fit(X, y) # 训练模型


# ### 取出决赛队伍的特征数据

# In[27]:


# 取出决赛队伍数据
final_team = pd.concat([home_team_df.loc['法国'].iloc[0], home_team_df.loc['克罗地亚'].iloc[0]])
final_team


# In[28]:


# 对数据进行归一化
final_team_normal = (final_team - play_score_temp.min()) / (play_score_temp.max() - play_score_temp.min())
final_team_normal


# ### 预测冠军球队【法国 🇫🇷 VS 克罗地亚 🇭🇷】

# In[29]:


model.predict(np.atleast_2d(final_team_normal)) # 预测


# 即代表法国取得冠军。
