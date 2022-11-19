##AWS Console:
##https://aws.amazon.com/console/
##
##Putty:
##https://www.chiark.greenend.org.uk/~sgtatham/putty/latest.html
##
##Boto3:
##https://boto3.amazonaws.com/v1/documentation/api/latest/index.html

import boto3

key_id = "AKIAV7Q4WIBCZQAYW47K"
secret_key = "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"

ec2 = boto3.client("ec2", aws_access_key_id = key_id,
                   aws_secret_access_key=secret_key,
                   region_name="eu-west-1")


##response = ec2.describe_regions()
##print('Regions:', response['Regions'])

response = ec2.describe_regions(Filters=[{"Name":"region-name","Values":["eu-west-1"]}])
print('Regions:', response['Regions'])

# Retrieves availability zones only for region of the ec2 object
response = ec2.describe_availability_zones()
print('Availability Zones:', response['AvailabilityZones'])


##Clients:
##   return description objects and appear lower level.
##   Description objects seem like AWS XML responses transformed
##   into Python Dicts/Lists.
##
##http://boto3.readthedocs.io/en/latest/reference/services/ec2.html?highlight=ec2.resource#client

##Resources:
##   return higher level Python objects like Instances with
##   stop/start methods.
##
##http://boto3.readthedocs.io/en/latest/guide/resources.html

