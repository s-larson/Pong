import gym
import numpy as np

def downsample(image):
    # Take only alternate pixels - basically halves the resolution of the image (which is fine for us)
    return image[::2, ::2, :]


def remove_color(image):
    """Convert all color (RGB is the third dimension in the image)"""
    return image[:, :, 0]


def remove_background(image):
    image[image == 144] = 0
    image[image == 109] = 0
    return image


def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))


def relu(vector):
    vector[vector < 0] = 0
    return vector


def preprocess_observations(input_observation, prev_processed_observation, input_dimensions):
    """ convert the 210x160x3 uint8 frame into a 6400 float vector """
    processed_observation = input_observation[35:195] # crop
    processed_observation = downsample(processed_observation)
    processed_observation = remove_color(processed_observation)
    processed_observation = remove_background(processed_observation)
    processed_observation[processed_observation != 0] = 1 # everything else (paddles, ball) just set to 1
    # Convert from 80 x 80 matrix to 1600 x 1 matrix
    processed_observation = processed_observation.astype(np.float).ravel()

    # subtract the previous frame from the current one so we are only processing on changes in the game
    if prev_processed_observation is not None:
        input_observation = processed_observation - prev_processed_observation
    else:
        input_observation = np.zeros(input_dimensions)
    # store the previous frame so we can subtract from it next time
    prev_processed_observations = processed_observation
    return input_observation, prev_processed_observations


def apply_neural_nets(observation_matrix, weights):
    """ Based on the observation_matrix and weights, compute the new hidden layer values and the new output layer values"""
    hidden_layer_values = np.dot(weights['1'], observation_matrix)
    hidden_layer_values = relu(hidden_layer_values)
    output_layer_values = np.dot(hidden_layer_values, weights['2'])
    output_layer_values = sigmoid(output_layer_values)
    return hidden_layer_values, output_layer_values


def choose_action(probability):
    random_value = np.random.uniform()
    if random_value < probability:
        # signifies up in openai gym
        return 2
    else:
         # signifies down in openai gym
        return 3


def main():
    env = gym.make("Pong-v0")
    observation = env.reset() # Gets image

    # Parameters
    episode_number = 0
    batch_size = 10 # how many rounds we play before updating the weights of our network
    gamma = 0.99 # the discount factor we use to discount the effect of old actions on the final result
    decay_rate = 0.99 # parameter used in RMSProp algorithm
    num_hidden_layer_neurons = 200 # how many neurons are in our hidden layer
    input_dimensions = 80 * 80 # dimension of our observation images
    learning_rate = 1e-4 # he rate at which we learn from our results to compute the new weights.
    # A higher rate means we react more to results and a lower rate means we don’t react as strongly to each result

    reward_sum = 0
    running_reward = None
    prev_processed_observations = None

    weights = {
        '1': np.random.randn(num_hidden_layer_neurons, input_dimensions) / np.sqrt(input_dimensions),
        '2': np.random.randn(num_hidden_layer_neurons) / np.sqrt(num_hidden_layer_neurons)
        # Weights are stored in matrixes. Layer 1 of our NN is a 200x6400 matrix representing the weights for
        # our hidden layer. Layer 2 is a 200x1 matrix representing the weights of the output of the hidden layer
        # on our final output. Atm random values divide by the square root of the number of the dimension size
    }

    # To be used with rmsprop algorithm (http://sebastianruder.com/optimizing-gradient-descent/index.html#rmsprop)
    expectation_g_squared = {}
    g_dict = {}
    for layer_name in weights.keys():
        expectation_g_squared[layer_name] = np.zeros_like(weights[layer_name])
        g_dict[layer_name] = np.zeros_like(weights[layer_name])

    episode_hidden_layer_values, episode_observations, episode_gradient_log_ps, episode_rewards = [], [], [], []
    # this stores observations and intermediate values across the episode to compute gradient

    while True:
        env.render()
        processed_observations, prev_processed_observations = preprocess_observations(observation,
                                                                                      prev_processed_observations,
                                                                                      input_dimensions)
        hidden_layer_values, up_probability = apply_neural_nets(processed_observations, weights)

        episode_observations.append(processed_observations)
        episode_hidden_layer_values.append(hidden_layer_values)

        action = choose_action(up_probability)

        # carry out the chosen action
        observation, reward, done, info = env.step(action)

        reward_sum += reward
        episode_rewards.append(reward)


main()
