# Determinsm Problems

## Synchronous mode in Carla
Carla Doku: https://carla.readthedocs.io/en/latest/adv_traffic_manager/#synchronous-mode

In Carla there are a Synchronous Mode and a Asynchronous Mode witch can set via the world settings.
`settings.synchronous_mode = True` [soruce](https://carla.readthedocs.io/en/latest/python_api/#carlaworldsettings). For the traffic manager there is also an async and a sync mode 
witch can be activated via traffic_manager.set_synchronous_mode(True).

According to Carla's documentation determinism can just achieve if the Traffic Manager and the Carla are 
set to Synchronous mode. But if both modes are set to Sync mode the simulation are still variate a bit each
execution. This can effect the behavior that much, that sometimes there is a collision and sometimes the car miss
the walker. 

## Differnt possivle cases 

To research the determinism problem I investigated into different possible possibilities.
That can be seen in the following table. For the test scenario I used an action sequence where the
walker run into the car. For the observation the first run where omitted because in the Async modes 
it varies a lot and for training the model if the first run varies a lot it isn't important. 

<table>
  <thead>
    <tr>
      <th>Case</th>
      <th>Carla</th>
      <th>Traffic Manaer</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>a)</td>
      <td>Yes </td>
      <td>Yes</td>
    </tr>
    <tr>
      <td>b)</td>
      <td>Yes</td>
      <td>No</td>
    </tr>
    <tr>
      <td>c)</td>
      <td>No</td>
      <td>Yes</td>
    </tr>
    <tr>
      <td>d)</td>
      <td>No</td>
      <td>No</td>
    </tr>
  </tbody>
</table>

a) At first the possibility sounds really promising for good deterministic results. 
But as mentioned before the results can variate a lot and in the end it is not usable for recurring and good results.

b) The best recurring results can be achieved with this method. 
But some results are still differ from the desired result. And it is also important that no waiting time is applied 
between each step or the results are differs like in the other combinations. 

c) By using this combination the environment crashes because, 
walker and car signals are sent to Carla, although these no longer exist. So this combination is not usable as well. 


d) As awful this combination sounds for determinism, as awful is it. The results are variate similar to a).
For the given action sequence there wasn't a situation that the difference matter that it decide of crash 
or no crash, but I'm almost certain that a situation like in the case of a) can be found. 

### Conglusion: 
For further action I will use Carla in Sync mode and the traffic manager in async mode because the results with 
this combination are the most repeatable and so probably for the training the best choice. 