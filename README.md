# Megaphone in Flink

We have tried to implement Megaphone as a application on top of Flink in which we avoided to hack on the source code.

As described in paper, the main challenge is to enable the state sharing among two continuous operators. 

In our implementation, we decided to use Kafka to enable the state sharing and also the frontier detection.

To enable operator F receiving reconfiguration from outside controllers, we chose to use Flink broadcast mechanism and implement a controller as an operator at the upstream of Operator F.

In our evaluation, we have found that Megaphone enables fluid state migration with very low latency, while the completion time for the state migration can be very high, the main challenge is coming from the synchronization between two operators i.e. frontier detection and state sharing.
