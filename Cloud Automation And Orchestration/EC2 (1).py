import boto3

class EC2Controller:
    # A class for controlling interactions with the boto3 EC2 Resource interface

    def __init__(self, ec2res):
        # EC2Controller Constructor, assigns the ec2 Resource "ec2res" to this controller
        self.ec2 = ec2res

    def stop_instance(self, instance_id):
        #Stop instance with id 'instance_id'

##        for instance in self.ec2.instances.all():
##            print(instance)
##            print(instance.instance_id)
##            #print(dir( instance ))
##            if instance.instance_id == instance_id:
##               instance.stop()
##               print("stopping instance", instance_id)

##        self.ec2.instances.filter(InstanceIds=[instance_id]).stop()

        
        self.ec2.Instance(instance_id).stop()

    def start_instance(self, instance_id):
        # Start instance with id 'instance_id'
        #self.ec2.instances.filter(InstanceIds=[instance_id]).start()

        self.ec2.Instance(instance_id).start()

    def list_instances(self):
        # List all EC2 instances

        count = 0
        # Loop through all EC2 instances
        for instance in self.ec2.instances.all():
            # Get a list of all tags on the instance
            tags = instance.tags

            # 'tags' is a list of tags, each of which is a dict in the format:
            # { "Key" : "keyvalue", "Value" : "valuevalue" }
            #print( tags )

            name = "Default EC2 Instance Name"
            for tag in tags:
                if tag["Key"] == 'Name':
                    name = tag['Value']

            # Output a description of the current instance
            print("Instance Name:", name, ", Instance Id:", instance.id,
                  ", State:", instance.state)

            count += 1

        #if(len( self.ec2.instances.all() ) == 0): #not valid!
        if count == 0:
            print("No EC2 Instances Detected!")

    def add_tags(self, instance_id, tags):
        # Add the contents of the list of dictionaries 'tags' as tags to
        # instance with id 'instance_id'
        self.ec2.Instance(instance_id).create_tags(Tags=tags)


    def delete_tags(self, instance_id, tags):
        # Delete the contents of the list of dictionaries 'tags' as tags from
        # instance with id 'instance_id'
        self.ec2.Instance(instance_id).delete_tags(Tags=tags)
            
    def create_instance(self, image_id):
        # Create a new EC2 instance from the given AMI 'image_id'
        self.ec2.create_instances(ImageId=image_id,InstanceType="t2.micro", MinCount=1, MaxCount=1)
 
