import boto3

class Volumes:
    # A class which functions as a controller for AWS EBS volumes

    def __init__(self, ec2res):
        # Volumes Constructor for  EC2 Resource "ec2res"
        self.ec2 = ec2res

    def list_volumes(self):
        # List all volumes associated with the with self.ec2 Resource

        
        count = 0
        for volume in self.ec2.volumes.all():
            #print(volume)

            print("***************************************************")

            print("Volume ID:", volume.volume_id)
            print("Volume State:", volume.state)
            print("Volume Size:",str(volume.size)+"GB")
            print("Volume Zone:", volume.availability_zone)
            print("Volume Type:", volume.volume_type)

            attach_data = volume.attachments
            #print(attach_data)
            for attachment in attach_data:
                print("EC2 Instance ID:", attachment["InstanceId"])
                print("Time of Attachment:", attachment["AttachTime"])
                print("Device:", attachment["Device"])

            print("***************************************************")

            count += 1

        if count == 0 :
            print("No EBS Volumes Detected!")

            
    def attach_volume(self, instance_id, volume_id, dev_name):

        # Attach volume with id "volume_id" to the EC2 instance with
        # id "instance_id", where it is the device "dev_name",
        # using the Resource "ec2"
        self.ec2.Instance(instance_id).attach_volume(VolumeId=volume_id,
                                                  Device=dev_name)
#How to attach a volume (note: device name!):
#http://boto3.readthedocs.io/en/latest/reference/services/ec2.html?highlight=attach_volume#EC2.Instance.attach_volume

#Alternative approach:
#http://boto3.readthedocs.io/en/latest/reference/services/ec2.html?highlight=attach_volume#EC2.Volume.attach_to_instance
#        ec2.Volume(volume_id).attach_to_instance(InstanceId=instance_id,Device=dev_name)
#

    def detach_volume(self, instance_id, volume_id, dev_name):
        # Detach the volume with id "volume_id" from the EC2 instance with
        # id "instance_id" where it is device "dev_name"
        self.ec2.Instance(instance_id).detach_volume(VolumeId=volume_id,
                                                  Device=dev_name)


#How to detach a volume:
#http://boto3.readthedocs.io/en/latest/reference/services/ec2.html?highlight=attach_vo

    def create_snapshot(self, volume_id, description):
        # Creates and returns a snapshot, with the given 'description',
        # of the EBS volume 'volume_id'.

        snapshot = self.ec2.create_snapshot(VolumeId=volume_id, Description=description)
        return snapshot


    def list_snapshots(self):
        # Prints out a list of all snapshots

        #ec2.snapshots.all() gets you ALL snapshots in existence (uh oh!)
        for snapshot in self.ec2.snapshots.filter(OwnerIds=["411303297093"]):
            print("Snapshot ID:", snapshot.snapshot_id)
            print("Snapshot Volume ID:", snapshot.volume_id)
            print("Snapshot Size:", snapshot.volume_size)
            print("Snapshot State:", snapshot.state)

    def get_snapshot(self, snapshot_id):
        # Returns the snapshot with the id 'snapshot_id', or None if it does
        # not exist.
        #snapshots = ec2.snapshots.filter(SnapshotIds=[snapshot_id])
        #print( snapshots.size ) #does not exist, exclude
        snapshot = self.ec2.Snapshot(snapshot_id)
        return snapshot

    def delete_snapshot(self, snapshot_id):

        # Deletes the snapshot with id 'snapshot_id'
        snapshot = self.ec2.Snapshot(snapshot_id)
        snapshot.delete()
    







    









        
