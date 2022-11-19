import resources
import EC2

res = resources.Resource()
ec2 = res.EC2Resource()

cont = EC2.EC2Controller(ec2)

#cont.stop_instance("i-0cb075fc2dbbfe8a6")

#cont.start_instance("i-0c4f012eb362844ec")

cont.list_instances()
