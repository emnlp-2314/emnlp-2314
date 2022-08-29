entity_sign = '▏'
role_sign = '▌'
event_sign = '▉'

a = 'incident type: attack▌ criminal individuals: high command▏ mid-level military officers▏ members of la tandona▏ col orlando cepeda▏ col rene emilio ponce▏ col guillermo benavides▏ general staff of the armed forces▏ alfredo cristiani▏ u.s. officers▏ advisers▌ criminal organizations: armed forces▌ physical targets:▌ human victims:▌ weapons:▉ incident type: robbery▌ criminal individuals: general staff of the armed forces▏ alfredo cristiani▏ u.s. officers▏ advisers▌ criminal organizations: armed forces▌ physical targets:'
event_bucket = a.split(event_sign + ' ')
print(event_bucket)
event_list = []
for i in range(len(event_bucket)):
    event_dict = {}
    role_bucket = event_bucket[i].split(role_sign + ' ')
    for j in range(len(role_bucket)):
        role_type = role_bucket[j].split(': ')[0]
        try:
            entity_bucket = role_bucket[j].split(': ')[1].split(entity_sign + ' ')
        except:
            entity_bucket = []
        event_dict[role_type] = entity_bucket
    event_list.append(event_dict)

print(event_list)
