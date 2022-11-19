import boto3

class EC2Controller:
    # A class for controlling interactions with the boto3 EC2 Resource interface

    def __init__(self, ec2res, region):
        # EC2Controller Constructor, assigns the ec2 Resource "ec2res" to this controller
        self.ec2 = ec2res
        self.region = region

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

    def list_instances(self, Print = True):
        # List all EC2 instances
        instancesDict = {}
        count = 0
        # Loop through all EC2 instances
        for instance in self.ec2.instances.all():
            # Get a list of all tags on the instance
            
            # 'tags' is a list of tags, each of which is a dict in the format:
            # { "Key" : "keyvalue", "Value" : "valuevalue" }
            #print( tags )
            name = "Default EC2 Instance Name"
            if instance.state["Code"] == 16:
                count += 1
                tags = instance.tags
                if tags:
                    for tag in tags:
                        if tag["Key"] == 'Name':
                            name = tag['Value']
                
                if Print:
                    print(f"\n############# Instance {count} #############\n\
                        \t- Instance Id: {instance.id}\n\
                        \t- Instance Name: {name}\n\
                        \t- State: {instance.state}\n\
                        \t- AMI Id: {instance.image_id}\n\
                        \t- Running since: {instance.launch_time}\n\
                        \t- Instance Type: {instance.instance_type}\n\
                        \t- Region: {self.region}")
                instancesDict[count] = {"Instance Id":instance.id,\
                                              "Name":name,\
                                              "AMI":instance.image_id,\
                                                  "State": instance.state["Name"]}
                
        for instance in self.ec2.instances.all():
            if instance.state["Code"] == 80:
                count += 1
                tags = instance.tags
                if tags:
                    for tag in tags:
                        if tag["Key"] == 'Name':
                            name = tag['Value']
                
                if Print:
                    print(f"\n############# Instance {count} #############\n\
                        \t- Instance Id: {instance.id}\n\
                        \t- Instance Name: {name}\n\
                        \t- State: {instance.state}\n\
                        \t- AMI Id: {instance.image_id}\n\
                        \t- Running since: {instance.launch_time}\n\
                        \t- Instance Type: {instance.instance_type}\n\
                        \t- Region: {self.region}")
                instancesDict[count] = {"Instance Id":instance.id,\
                                              "Name":name,\
                                              "AMI":instance.image_id,\
                                                  "State": instance.state["Name"]}
                
        
        for instance in self.ec2.instances.all():
            if instance.state["Code"] in [0,32,48,64]:
                count += 1
                tags = instance.tags
                if tags:
                    for tag in tags:
                        if tag["Key"] == 'Name':
                            name = tag['Value']
                
                if Print:
                    print(f"\n############# Instance {count} #############\n\
                        \t- Instance Id: {instance.id}\n\
                        \t- Instance Name: {name}\n\
                        \t- State: {instance.state}\n\
                        \t- AMI Id: {instance.image_id}\n\
                        \t- Running since: {instance.launch_time}\n\
                        \t- Instance Type: {instance.instance_type}\n\
                        \t- Region: {self.region}")
                instancesDict[count] = {"Instance Id":instance.id,\
                                              "Name":name,\
                                              "AMI":instance.image_id,\
                                                  "State": instance.state["Name"]}
                
        #if(len( self.ec2.instances.all() ) == 0): #not valid!
        if count == 0:
            print("No EC2 Instances Detected!")
            
        return instancesDict
            
