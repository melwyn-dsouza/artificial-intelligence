import resources
import EC2
import Volumes
import S3
import Cloudwatch

res = resources.Resource()
ec2 = res.EC2Resource()
s3 = res.S3Resource()
cw = res.CWClient()

cont = EC2.EC2Controller(ec2)
vol = Volumes.Volumes(ec2)
s3cont = S3.S3Controller(s3)
contCW = Cloudwatch.CWController(cw)

#cont.stop_instance("i-0cb075fc2dbbfe8a6")

#cont.start_instance("i-0c4f012eb362844ec")

#cont.list_instances()

#tags=[{"Key":"key1","Value":"value1"},{"Key":"key2","Value":"value2"}]


#cont.add_tags("i-08b6f1c41e6b907cc", tags)

#cont.delete_tags("i-08b6f1c41e6b907cc", tags)

#cont.create_instance("ami-0fd8802f94ed1c969")

#vol.list_volumes()

#vol.detach_volume("i-08b6f1c41e6b907cc", "vol-012327535dfe48137", "/dev/sdf")

#vol.attach_volume("i-0ed377696069e4e8c", "vol-012327535dfe48137", "/dev/sdf") 

#snap = vol.create_snapshot("vol-012327535dfe48137", "Snappy 8th Nov")

#vol.list_snapshots()

#vol.delete_snapshot("snap-05d00c4f02d91fa8f")

#s3cont.upload_file("stynes2022lecture", "hello.txt", "diffName")
#s3cont.download_file("stynes2022lecture", "diffName", "grd89sge.txt")

#s3cont.delete_bucket("stynes2022lecture")

#contCW.get_metric_statistics("i-08b6f1c41e6b907cc", "CPUUtilization")

#contCW.set_alarm("i-08b6f1c41e6b907cc", "CPUUtilization", 75.0, "Percent")

contCW.delete_alarm("Alarm Example!") 

