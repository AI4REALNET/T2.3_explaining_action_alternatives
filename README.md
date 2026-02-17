# Explanation of Action Alternative Outcomes

This repository implements the explanation of action alternatives described in AI4RealNet deliverable D2.3.

We consider a simple control setting where an AI generates one or more explanations for each action a (human) operator can take in the current state of a control problem. Following existing work in explainable-AI, we consider explanations that explain what outcomes (defined as discounted cumulative rewards or features) will be achieved under each action. What is special is that, unlike prior work, in our setting the AI does not know how the operator weights these outcomes in their reward function, and thus does not know what policy the operator follows. Despite this, we want to be able to guarantee that outcomes are accurate, in the sense that outcomes will be realized in expectation. To do this we introduce an AI that maintains beliefs about the operator's reward weights and learn to predict _expected_ outcomes under this evolving belief. Please consult deliverable D2.3 for a complete description of the method.

## Requirements

The implementation in this repository uses Python with a number of external software libraries. Control problems are implemented using the Gymnasium package and Numpy. The machine learning of team outcomes is implemented using PyTorch. Please refer to `requirements.txt` for a full list of dependencies.

## Overview of Code Structure

├── **envs**: Control problem implementations.  
│   &ensp;&ensp;&ensp;&ensp;└── `dam.py`: Dam control environment.  
│   &ensp;&ensp;&ensp;&ensp;└── `mo_gridworld.py`: GridWorld environment.  
├── **team_SF**:   
│   &ensp;&ensp;&ensp;&ensp;└── `explanation_task.py`: Implementation of interaction between operator and AI.  
│   &ensp;&ensp;&ensp;&ensp;└── `models.py`: PyTorch models for outcome prediction.  
│   &ensp;&ensp;&ensp;&ensp;└── `train_outcomes.py`: Learning code for outcome prediction.  
│   &ensp;&ensp;&ensp;&ensp;└── `trajectory_buffer.py`: Utilities for outcome prediction.  
├── `main.py`

## Training and Testing

To run the code, install the dependecies defined in `requirements.txt` and run:

```bash
python main.py --env $ENV
```

Where `$ENV` is one of:
- `MOGridWorld`: A toy grid world problem designed specifically to test outcome prediction in teams.
- `Dam`: A realistic Dam control problem from Castelletti et al. [DOI: 10.1109/IJCNN.2012.6252759].

The above command will train a neural network to predict outcomes for a given operator whose true reward parameters it does not know. Once trained, the resulting predictor will be tested and two graphs will be displayed: One showing how the AI's uncertainty about the operator's reward weights evolves over time, and one showing the error between the Predictor's outcome predictions and the empirically realizes outcomes.
