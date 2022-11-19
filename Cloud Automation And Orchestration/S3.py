import boto3

class S3Controller:

    #A class for controlling interactions with amazon's S3 service
    def __init__(self, s3res):
        #S3Controller Constructor

        self.s3 = s3res

#https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/s3.html?highlight=s3#s3

    def list_buckets(self):

        # Print out a list of all s3 buckets your credentials have created

        for bucket in self.s3.buckets.all():
            print(type(bucket))

    def create_bucket(self, bucket_name):

        # Create and return S3 bucket with name 'bucket_name'
        self.s3.create_bucket( Bucket=bucket_name, CreateBucketConfiguration={
                    'LocationConstraint': 'eu-west-1'})
        return self.s3.Bucket( bucket_name )

#creating a bucket
#http://boto3.readthedocs.io/en/latest/reference/services/s3.html?highlight=s3.create_bucket#S3.Client.create_bucket

    def delete_bucket(self, bucket_name):

        # Delete the S3 bucket with name 'bucket_name'

        bucket = self.s3.Bucket(bucket_name)
        #must delete all objects/keys before you can delete a bucket
        for key in bucket.objects.all():
            key.delete()

        bucket.delete()


    def upload_file(self, bucket_name, file_name, key):
        # Upload the file 'file_name' to S3 storage, into the bucket
        # 'bucket_name'. The name 'key' will
        # be used to reference the file in the S3 storage

        bucket = self.s3.Bucket(bucket_name)
        bucket.upload_file(file_name, key)

    def download_file(self, bucket_name, key, local_file_name):

        # Download the file referenced by 'key' in the S3 bucket with
        # name 'bucket_name', to the file location 'local_file_name'

        self.s3.Bucket(bucket_name).download_file(key, local_file_name)

    def delete_file(self, bucket_name, key):

        # Delete the file with key 'key' from S3 storage, from the bucket
        # 'bucket_name'

        bucket = self.s3.Bucket(bucket_name)
        bucket.delete_objects(
            Delete={
                "Objects":[
                    {
                        "Key":key
                    }
                ]
            }
        )


