import boto3

class Resource:
    #A class for creating boto3 "Resource" interfaces to AWS services

    def __init__(self):
        #Resource instance

        self.region = "eu-west-1"
        self.key_id = "AKIAV7Q4WIBCZQAYW47K"
        self.secret_key = "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"

    def EC2Resource(self):
        # Create and return a Resource for interacting with EC2 instances
        ec2 = boto3.resource("ec2",aws_access_key_id=self.key_id,
                     aws_secret_access_key=self.secret_key,
                     region_name=self.region)
        return ec2

    def S3Resource(self):
        # Create and return a Resource for interacting with S3 instances
        s3 = boto3.resource("s3",aws_access_key_id=self.key_id,
                     aws_secret_access_key=self.secret_key,
                     region_name=self.region)
        return s3

    def CWClient(self):
        # Create and return a Client for interacting with CloudWatch
        cw = cloudwatch = boto3.client('cloudwatch',aws_access_key_id=self.key_id,
                     aws_secret_access_key=self.secret_key,
                     region_name=self.region)
        return cw
