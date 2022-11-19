import boto3
import datetime

class CWController:

    # A class for controlling AWS CloudWatch settings to monitor metrics
    # and create/use alarms.

    def __init__(self , cwRes):

        # CWController Constructor
        self.cw = cwRes

    def get_metric_statistics(self, instance_id, metric):
        # Output the average result of the given 'metric' over the last 600 seconds
        # for EC2 instance 'instance_id'

        a = self.cw.get_metric_statistics(
            Period=300,
            StartTime=datetime.datetime.utcnow() - datetime.timedelta(seconds=600),
            EndTime=datetime.datetime.utcnow(),
            MetricName=metric,
            Namespace="AWS/EC2",
            Statistics=['Average'],
            Dimensions=[{'Name':'InstanceId', 'Value':instance_id}]
            )
        print( a )

    def set_alarm(self, instance_id, metric, value, unit_type):

        # Create alarm for the instance 'instance_id' on the metric 'metric'
        # to trigger if it exceeds threshold 'value'. The 'unit_type' specifies
        # the units used by the given metric

        self.cw.put_metric_alarm(
            AlarmName='Alarm Example!',
            ComparisonOperator='GreaterThanThreshold',
            EvaluationPeriods=1,
            MetricName=metric,
            Namespace='AWS/EC2',
            Period=300,    #INSUFFICIENT_DATA error if lower than the metric period
            Statistic='Average',
            Threshold=value,
            ActionsEnabled=False,
            AlarmDescription='Alarm example for Cloud Computing with Python!',
            Dimensions=[
                {
                    'Name': 'InstanceId',
                    'Value': instance_id
                },
            ],
            Unit=unit_type
            )


    def delete_alarm(self, name):

        # Deletes an alarm with the given 'name'
        self.cw.delete_alarms(
            AlarmNames=[name]
            )
        
        
