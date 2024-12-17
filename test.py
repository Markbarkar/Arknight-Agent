import json

data = {}
res = {}
player_list = ['维什戴尔', '泡普卡', '史都华德', '苏苏洛', '卡提', '克洛斯', '桃金娘', '芬']
high_player_list = ['维什戴尔', '史都华德', '苏苏洛', '克洛斯']
high_floor_list_id = [1, 4, 8, 10, 12, 13, 15, 17]

player_fee_list = [25, 17, 16, 16, 16, 9, 8, 9]
position_location_list = [(1439, 405), (1420, 266), (1289, 392), (1471, 529),
                          (1281, 261), (1130, 389), (1321, 519), (1502, 671),
                          (1137, 267), (985, 379), (1160, 520), (1338, 653),
                          (1545, 823), (998, 263), (841, 392), (992, 518), (1172, 681), (1365, 819)]

for t, i in enumerate(player_list):
    ishigh = 1 if i in high_player_list else 0
    fee = player_fee_list[t]
    freeze_time = 0
    skill_time = 0
    res[i] = {
        'ishigh': ishigh,
        'fee': fee,
        'freeze_time': freeze_time,
        'skill_time': skill_time
    }

data.update(players=res, position_location_list=position_location_list, action_time=1, protect_point=3,
            high_floor_list_id=high_floor_list_id)
data_json = json.dumps(data, ensure_ascii=False)
file = open("data.json", 'w', encoding="utf8")
file.write(data_json)
file.close()

# with open("data.json", 'r', encoding='utf-8') as f:
#     data = json.loads(f.read())
#     print([i['fee'] for i in data['players'].values()])
