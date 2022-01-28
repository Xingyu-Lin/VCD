# Chester

Chester is a tool aiming at automatically launching experiments. This tool based on rllab(https://github.com/rll/rllab ), and further extended for launching and retrieving experiments in different remote machines, including:
1. Seuss
2. PSC (Pittsburgh Super Computing)

## Getting Started

We've provided an example for launching experiments of openai/baseline's DDPG algorithm.

Look into the /examples, you'll find 'train_luanch.py' and 'train.py'. 'train.py' is the parser where we copied a lot of codes in openai/baseline/ddpg/main.py and combined them as a function 'run_task'. 'run_task' receives the parameters and start running the DDPG algorithm with those given settings. 

The launcher 'train_launch.py' uses our chester and the 'run_task' function to launch a group of experiments locally. By running this launcher, the group experiments are started and the resutls are contained in one given folder. Those result files are able to be visulized with rllab's viskit. 

To support different options in visulization, chester provided self-written interface 'preset.py'. The author can write different custom splitters in this file and put it in the directory for experiments. The viskit tool can detect this preset file and apply different options. 


### Prerequisites

What things you need to install the software and how to install them

```
Give examples
```

### Installation 

A step by step series of examples that tell you how to get a development env running

Say what the step will be

```
Give the example
```

And repeat

```
until finished
```

End with an example of getting some data out of the system or using it for a little demo

## Running the tests

Explain how to run the automated tests for this system

### Break down into end to end tests

Explain what these tests test and why

```
Give an example
```

### And coding style tests

Explain what these tests test and why

```
Give an example
```

## Deployment

Add additional notes about how to deploy this on a live system

## Built With

* [Dropwizard](http://www.dropwizard.io/1.0.2/docs/) - The web framework used
* [Maven](https://maven.apache.org/) - Dependency Management
* [ROME](https://rometools.github.io/rome/) - Used to generate RSS Feeds

## Contributing

Please read [CONTRIBUTING.md](https://gist.github.com/PurpleBooth/b24679402957c63ec426) for details on our code of conduct, and the process for submitting pull requests to us.

## Versioning

We use [SemVer](http://semver.org/) for versioning. For the versions available, see the [tags on this repository](https://github.com/your/project/tags). 

## Authors

* **Billie Thompson** - *Initial work* - [PurpleBooth](https://github.com/PurpleBooth)

See also the list of [contributors](https://github.com/your/project/contributors) who participated in this project.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Hat tip to anyone whose code was used
* Inspiration
* etc

