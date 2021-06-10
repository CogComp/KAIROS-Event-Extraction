from util import *

all_role_types = list()
for tmp_role in role_to_type:
    all_role_types += role_to_type[tmp_role]

all_role_types = list(set(all_role_types))

ontonotes_types = ['per', 'gpe', 'org', 'fac', 'loc', 'wea', 'veh', 'com', 'nat', 'mhi', 'val',
                   'law', 'abd', 'bod', 'mon', 'sid', 'pth', 'ttl', 'inf']

for tmp_type in all_role_types:
    if tmp_type not in ontonotes_types:
        print(tmp_type)

print('end')