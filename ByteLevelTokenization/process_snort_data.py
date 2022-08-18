import json
import os
import re
import numpy as np


def process_snort(file_name):
    # print(os.path.abspath(__file__))
    file_path = os.path.dirname(__file__) + "/data/snort/rules/emerging-{}.rules".format(file_name)
    output_path = os.path.dirname(__file__) + "/data/snort/rules/emerging-{}.txt".format(file_name)
    content_rule_path = os.path.dirname(__file__) + "/data/snort/rules/emerging-content-{}.txt".format(file_name)

    content_rule = []
    contents = []
    rules = {}
    idx = 0
    with open(file_path, "r") as f:

        for line in f:
            content = re.search(r'content:"[^"]*";', line)
            if content != None:
                content = content.group()[9:-2]
                if any(content == s for s in contents) or len(content) < 2:
                    continue

                contents.append(content)
                # print(content)
                content_rule.append(line)
                channelFlag = False
                rule = ['$', '*']
                i = 0
                while i < len(content):
                    if content[i] != '|':
                        if channelFlag == False:
                            rule.append(str(hex(ord(content[i]))))
                            i = i + 1
                        else:
                            # print(content[i:i + 2])

                            binNumber = '0x' + content[i:i + 2].lower()
                            rule.append(binNumber)

                            i = i + 2
                            if content[i] == '|':
                                i = i + 1
                                channelFlag = False
                                continue

                            if content[i].isspace():
                                i = i + 1
                    else:
                        i = i + 1
                        channelFlag = not channelFlag

                if len(rule) < 4:
                    continue
                rule.append('$')
                rule.append('*')
                rules[idx] = rule
                idx = idx + 1

    with open(output_path, "w") as f:
        f.write(json.dumps(rules))

    with open(content_rule_path, "w") as f:
        for rule in content_rule:
            f.write(rule)

    print("{} Finished!".format(file_name))


if __name__ == "__main__":
    snort_dataset = ['activex', 'attack_response', 'botcc.portgrouped', 'botcc', 'chat', 'ciarmy', 'compromised',
                     'current_events', 'deleted', 'exploit', 'ftp', 'games', 'icmp', 'icmp_info',
                     'imap', 'inappropriate', 'info', 'malware', 'misc',
                     'mobile_malware', 'netbios', 'p2p', 'policy', 'pop3', 'rpc', 'scada', 'scan', 'shellcode', 'smtp',
                     'snmp', 'sql', 'telnet', 'tftp', 'tor', 'trojan', 'user_agents', 'voip', 'web_client',
                     'web_server', 'web_specific_apps', 'worm']
    # snort_dataset = ['games']
    for snort in snort_dataset:
        process_snort(snort)
