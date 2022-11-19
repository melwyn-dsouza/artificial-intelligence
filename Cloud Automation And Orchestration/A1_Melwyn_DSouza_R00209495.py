# -*- coding: utf-8 -*-
"""
Created on Tue Nov 15 11:52:36 2022

@author: dsouzm3
"""

import sys
import resources
import EC2

Access_Key_ID = "AKIA2GDSSSTM23X4D66T"
Secret_Access_Key = "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"

# pw_file = open("passwords.txt", 'r+')

# pw_dict = {}
# for line in pw_file.readlines():
#     l = line.split("\t")
#     pw_dict[l[0].strip()] = [x.strip() for x in l[1:]]

    
# print(pw_dict) 
# mode = int(input("Select\n1 - Login\n2 - Register(New User)\n3 - Exit\n>>"))

# if (mode == 3):
#     sys.exit(0)

# while(True):
#     if mode == 1:
#         username = input("Enter Username:\n>>")
#         if username in pw_dict.keys():
#             password = input("Enter Password:\n>>")
#             if password in pw_dict[username]:
#                 print("Logged in successfully")
#                 break
#             elif password == '':
#                 print("Blank password, exiting...")
#                 pw_file.close()
#                 sys.exit(0)
#             else:
#                 print("Wrong password, please retry!!")
#         elif username == '':
#             print("Blank username, exiting...")
#             pw_file.close()
#             sys.exit(0)
#         else:
#             print("User does not exist, please retry!!")
    
#     elif mode == 2:
#         admin = int(input("Select\n1 - Admin\n2 - Regular(New User)\n3 - Exit\n>>"))
#         username = input("Enter Username:\n>>")
#         if username in pw_dict.keys():
#             print("Username already exists, please change and retry!!")
#             continue
#         elif username == '':
#             print("Blank username, exiting...")
#             sys.exit(0)
        
#         password = input("Enter Password:\n>>")
        
#         if password == '':
#             print("Blank password, exiting...")
#             pw_file.close()
#             sys.exit(0)
#         else:
#             break

# if mode == 2:
#     if admin == 1:
#         pw_file.write(f"\n{username}\t{password}\t{Access_Key_ID}\t{Secret_Access_Key}\tadmin")
#         pw_dict[username] = [password, Access_Key_ID, Secret_Access_Key, "admin"]
#     elif admin == 2:
#         pw_file.write(f"\n{username}\t{password}\t{Access_Key_ID}\t{Secret_Access_Key}\tregular")
#         pw_dict[username] = [password, Access_Key_ID, Secret_Access_Key, "regular"]
        
# pw_file.close()


while True:
    service = int(input("Select\n1 - EC2\n2 - EBS\n3 - S3\n>>"))
    
    while True:
        if service ==  1:
            res = resources.Resource(Access_Key_ID, Secret_Access_Key)
            ec2, region = res.EC2Resource()
            cont = EC2.EC2Controller(ec2, region)   
            
            ec2_menu = int(input("Select\n1 - List Instances\n2 - Start Instance\n3 - Stop Instance\n4 - Create AMI from Instance\n5 - Delete AMI\n6 - Back to Main Menu\n7 - Exit\n>>"))
            
            if ec2_menu == 1:    
                cont.list_instances()
                
            elif ec2_menu == 2:
                instances = cont.list_instances(Print=False)
                print("\nSTART INSTANCE: Listing all instances")
                for k,v in instances.items():
                    print(f"{k}: Name: {v['Name']}, ID: {v['Instance Id']}, State: {v['State']}")
                start_inst = int(input("Select Instance to start:\n>>"))
                
                if start_inst in instances.keys():
                    if instances[start_inst]['State'] == "running":
                        print(f"Instance {instances[start_inst]['Instance Id']} is already running, retry..")
                        continue
                    else:
                        print(f"\nStarting {instances[start_inst]['Instance Id']}")
                        cont.start_instance(instances[start_inst]['Instance Id'])
                    
            elif ec2_menu == 3:
                instances = cont.list_instances(Print=False)
                print("\nSTOP INSTANCE: Listing all instances")
                for k,v in instances.items():
                    print(f"{k}: Name: {v['Name']}, ID: {v['Instance Id']}, State: {v['State']}")
                stop_inst = int(input("Select Instance to stop:\n>>"))
                
                if stop_inst in instances.keys():
                    if instances[stop_inst]['State'] == "stopped":
                        print(f"Instance {instances[stop_inst]['Instance Id']} is already stopped, retry..")
                        continue
                    else:
                        print(f"\nStopping {instances[stop_inst]['Instance Id']}")
                        cont.stop_instance(instances[stop_inst]['Instance Id'])
                
            elif ec2_menu == 4:
                print("Create AMI")
            
            elif ec2_menu == 5:
                print("Delete AMI")
            
            elif ec2_menu ==6:
                print("Back to main")
                break
            
            elif ec2_menu == 7:
                sys.exit(0)
                


