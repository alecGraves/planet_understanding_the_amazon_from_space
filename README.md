# planet_understanding_the_amazon_from_space
My solution to the caggle competitionPlanet: understanding the amazon from space.
* Final rank ~= 300/800, .92 Fscore2


## Notes
* Simplenet seems to offer great performance for the size
* incepnet is definitely the best model as of now
* darknet offers about the same performance as simplenet
* alleluna seems to score below simplenet and overfitts over time
* incepnet_v4 seems to be terrible at this and really slow to train
* simplenet xxl is untested, seemed to offer decent performance, but overfit (i think?)
* simplenet_v2 and simplenet_v2_m are being tested

## TODO (que)
* make a performance evaluation function, see most missed class, etc
* create special networks trained on hard classes
    * try cirriculum learning?
* create ensemble conductor network
