#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

// Auxiliaries functions
double randowm_weight()
{
  return ((double)rand() / RAND_MAX) - 0.5;
}

typedef struct
{
  double *weigth;
  double bias;
  double output;
  double gradient;
  double delta;
  int num_of_inputs;
} Neuron;

typedef struct
{
  Neuron *neurons;
  int num_of_neurons;
} Layer;

typedef struct
{
  Layer *layers;
  int num_of_layers;
  double (*activation_funciton)(double);
  double (*derivative_activation)(double);
} NeuralNetwork;

// Activation Functions

double sigmoid(double x)
{
  return 1.0 / (1.0 + exp(-x));
}

double derivative_sigmoid(double x)
{
  return x * (1.0 - x);
}

double relu(double x)
{
  return x > 0 ? x : 0;
}

double derivative_relu(double x)
{
  return x > 0 ? 1 : 0;
}

// Network Management Functions

NeuralNetwork *create_network(int num_of_layers, int neurons_by_layer[], double (*activation)(double), double (*derivative)(double))
{
  NeuralNetwork *network = (NeuralNetwork *)malloc(sizeof(NeuralNetwork));
  network->num_of_layers = num_of_layers;
  network->activation_funciton = activation;
  network->derivative_activation = derivative;

  network->layers = (Layer *)malloc(num_of_layers * sizeof(Layer));

  for (int i = 0; i < num_of_layers; i++)
  {
    network->layers[i].num_of_neurons = neurons_by_layer[i];

    network->layers[i].neurons = (Neuron *)malloc(neurons_by_layer[i] * sizeof(Neuron));

    for (int j = 0; j < neurons_by_layer[i]; j++)
    {
      Neuron *neuron = &network->layers[i].neurons[j];

      int num_inputs_for_neuron = (i == 0) ? 0 : neurons_by_layer[i - 1];
      neuron->num_of_inputs = num_inputs_for_neuron;

      neuron->bias = 0;

      if (num_inputs_for_neuron > 0)
      {
        neuron->weigth = (double *)malloc(num_inputs_for_neuron * sizeof(double));

        for (int k = 0; k < num_inputs_for_neuron; k++)
        {
          neuron->weigth[k] = randowm_weight();
        }
      }
      else
      {
        neuron->weigth = NULL;
      }
    }
  }
  return network;
}

void release_network(NeuralNetwork *network)
{
  for (int i = 0; i < network->num_of_layers; i++)
  {
    for (int j = 0; j < network->layers[i].num_of_neurons; j++)
    {
      if (network->layers[i].neurons[j].weigth != NULL)
      {
        free(network->layers[i].neurons[j].weigth);
      }
    }
    free(network->layers[i].neurons);
  }
  free(network->layers);
  free(network);
}

// Forward Propagation Functions

double activate_neuron(Neuron *neuron, double *inputs)
{
  double activation_sum = neuron->bias;

  for (int i = 0; i < neuron->num_of_inputs; i++)
  {
    activation_sum += neuron->weigth[i] * inputs[i];
  }
  return activation_sum;
}

double *forward_propagate(NeuralNetwork *network, double *inputs)
{
  double *current_inputs = inputs;
  for (int i = 0; i < network->num_of_layers; i++)
  {
    Layer *current_layer = &network->layers[i];
    double *layer_outputs = (double *)malloc(current_layer->num_of_neurons * sizeof(double));
    for (int j = 0; j < current_layer->num_of_neurons; j++)
    {
      Neuron *current_neuron = &current_layer->neurons[j];
      if (i == 0)
      {
        current_neuron->output = current_inputs[j];
      }
      else
      {
        double z = activate_neuron(current_neuron, current_inputs);
        current_neuron->output = network->activation_funciton(z);
      }
      layer_outputs[j] = current_neuron->output;
    }
    if (i > 0)
    {
      free(current_inputs);
    }
    current_inputs = layer_outputs;
  }
  return current_inputs;
}

// Backpropagation and Training Functions

void calculate_delta_output(NeuralNetwork *network, double *expected)
{
  Layer *output_layer = &network->layers[network->num_of_layers - 1];

  for (int i = 0; i < output_layer->num_of_neurons; i++)
  {
    Neuron *neuron = &output_layer->neurons[i];

    double error = neuron->output - expected[i];

    neuron->delta = error * network->derivative_activation(neuron->output);
  }
}

void propagate_error_backwards(NeuralNetwork *network)
{
  for (int i = network->num_of_layers - 2; i > 0; i--)
  {
    Layer *current_layer = &network->layers[i];
    Layer *next_layer = &network->layers[i + 1];
    for (int j = 0; j < current_layer->num_of_neurons; j++)
    {
      Neuron *neuron = &current_layer->neurons[j];
      double error_sum = 0.0;
      for (int k = 0; k < next_layer->num_of_neurons; k++)
      {
        error_sum += next_layer->neurons[k].weigth[j] * next_layer->neurons[k].delta;
      }
      neuron->delta = error_sum * network->derivative_activation(neuron->output);
    }
  }
}

void update_parameters(NeuralNetwork *network, double *inputs, double learning_rate)
{
  for (int i = 1; i < network->num_of_layers; i++)
  {
    Layer *current_layer = &network->layers[i];
    Layer *previous_layer = &network->layers[i - 1];

    double *current_inputs = (i == 1) ? inputs : NULL;

    for (int j = 0; j < current_layer->num_of_neurons; j++)
    {
      Neuron *neuron = &current_layer->neurons[j];

      for (int k = 0; k < neuron->num_of_inputs; k++)
      {
        double input_for_weigth = (i == 1) ? current_inputs[k] : previous_layer->neurons[k].output;

        neuron->weigth[k] -= learning_rate * neuron->delta * input_for_weigth;
      }
      neuron->bias -= learning_rate * neuron->delta;
    }
  }
}

// Main Training Role

void train_network(NeuralNetwork *network, double **training_data, double **expected, int num_of_samples, int epochs, double learning_rate)
{
  for (int i = 0; i < epochs; i++)
  {
    double epoch_error = 0.0;
    for (int j = 0; j < num_of_samples; j++)
    {
      double *outputs = forward_propagate(network, training_data[j]);

      for (int k = 0; k < network->layers[network->num_of_layers - 1].num_of_neurons; k++)
      {
        epoch_error += pow(expected[j][k] - outputs[k], 2);
      }

      calculate_delta_output(network, expected[j]);
      propagate_error_backwards(network);

      update_parameters(network, training_data[j], learning_rate);

      free(outputs);
    }
    if ((i + 1) % 1000 == 0)
    {
      printf("Epoch %d/%d, Error=%.6f\n", i + 1, epochs, epoch_error / num_of_samples);
    }
  }
}

// C Main Function

int main(int argc, char *argv[])
{
  srand(time(NULL));

  int topology[] = {2, 2, 1};
  int num_layers = sizeof(topology) / sizeof(int);

  double training_inputs[4][2] = {{0.0, 0.0}, {0.0, 1.0}, {1.0, 0.0}, {1.0, 1.0}};
  double expected_outputs[4][1] = {{0.0}, {1.0}, {1.0}, {0.0}};

  double *p_training_inputs[4];
  double *p_expected_outputs[4];
  for (int i = 0; i < 4; i++)
  {
    p_training_inputs[i] = training_inputs[i];
    p_expected_outputs[i] = expected_outputs[i];
  }

  NeuralNetwork *net = create_network(num_layers, topology, sigmoid, derivative_sigmoid);

  printf("Starting training...\n");
  train_network(net, p_training_inputs, p_expected_outputs, 4, 10000, 0.1);
  printf("Training completed.\n\n");

  printf("Testing the trained network:\n");
  for (int i = 0; i < 4; i++)
  {
    double *prediction = forward_propagate(net, p_training_inputs[i]);
    printf("Input: [%.1f, %.1f] -> Prediction: %.6f (Expected: %.1f)\n",
           p_training_inputs[i][0], p_training_inputs[i][1], prediction[0], p_expected_outputs[i][0]);
    free(prediction);
  }

  release_network(net);
  printf("\nFeatures released.\n");

  return 0;
}
