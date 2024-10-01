using NumSharp;
using System;
using System.Linq;
using System.Threading.Tasks;
using System.Xml;

namespace NNSubroutinesReal
{
    internal class Program
    {
        static void Main(string[] args)
        {
            NDArray x_train;
            NDArray y_train;
            /*
            Runtime.PythonDLL = @"C:\Users\zrizw\AppData\Local\Programs\Python\Python312\python312.dll";
            using (Py.GIL()) //load mnist using python
            {
                var ((train_images, train_labels), (test_images, test_labels)) = Keras.Datasets.MNIST.LoadData();
                trainImagesArray = np.zeros(train_images.shape[0], train_images.shape[1]);
                trainImagesArray.SetData(train_images);
                trainLabelsArray = np.zeros(train_labels.shape[0], train_labels.shape[1]);
                trainLabelsArray.SetData(train_labels);
            }
            PythonEngine.Shutdown();
            
            //recalibrate arrays for nn
            NDArray x_train = trainImagesArray.reshape((60000, 784));
            x_train = x_train.astype(np.float32) / 255;
            NDArray y_train = np.zeros(trainImagesArray.size, 10);
            for (int i = 0; i < trainImagesArray.size; i++)
            {
                y_train[i, trainLabelsArray[i]] = 1;
            }
            */

            //create and print training data
            x_train = np.array([[0,0],
            [0,1],
            [1,0],
            [1,1]]);
            y_train = np.array([0, 1, 1, 0]).reshape(4, 1);
            Console.WriteLine(x_train.ToString());
            Console.WriteLine(y_train.ToString());
            
            //create and train model
            CreateModel nn = new CreateModel(2, 1, [2, 2]);
            nn.train(x_train, y_train, 100, 0.1f, 0.001f);
            Console.Read();
        }
    }

    /* Implements a full NN in one class. a synapse is a connection between 2 neurons so synapse value is just weight -_-
    Incomplete - this implementation is not applicable to NEA project as uses Sigmoid/Sigmoid derivative with no explanations of how to implement more complex stuff (ie Adam MSE etc.) */
    class NN
    {
        private Random RNGesus;
        public NN(int synapseMatrixColumns, int synapseMatrixLines)
        {
            SynapseMatrixColumns = synapseMatrixColumns;
            SynapseMatrixLines = synapseMatrixLines;

            _Init();
        }

        public int SynapseMatrixColumns { get; }
        public int SynapseMatrixLines { get; }
        public double[,] SynapseMatrix { get; private set; }

        /* Initalise random object and matrix of random weights. */
        private void _Init()
        {
            RNGesus = new Random();
            _GenerateSynapseMatrix();
        }

        /* Generate matric with weight of synapses (randomly - there are other methods for this though) */
        private void _GenerateSynapseMatrix()
        {
            SynapseMatrix = new double[SynapseMatrixLines, SynapseMatrixColumns];
            for (int i = 0; i < SynapseMatrixLines; i++)
            {
                for (int j = 0; j < SynapseMatrixColumns; j++)
                {
                    SynapseMatrix[i, j] = (2 * RNGesus.NextDouble()) - 1;  //to give -ve and +ve values between 0 and 1
                }
            }
        }

    }

    /* Implements MLP with Adam and other cool features, by using one class for a layer and another to fully construct the network. 
     * this is useful because we may require multiple different types of NN (ie A2N, PPO strategies*/
    class FCLayer
    {
        public FCLayer(int input_size_input, int output_size_input, string activation_input)
        {
            input_size = input_size_input;
            output_size = output_size_input;
            activation = activation_input;

            _Init();
        }
        public int input_size { get; }
        public int output_size { get; }
        public string activation { get; }
        //regular weights and biases
        public NDArray weights;
        private NDArray biases;
        //derivatives of weights and biases
        public NDArray d_weights;
        private NDArray d_biases;
        /* Define m & v for weights and biases
         * These variables are used in Adam optimisation */
        private NDArray m_weights;
        private NDArray m_biases;
        private NDArray v_weights;
        private NDArray v_biases;
        private NDArray m_hat_biases;
        private NDArray v_hat_biases;
        private NDArray m_hat_weights;
        private NDArray v_hat_weights;
        //hyperparameters for Adam - ADD AS PARAMETERS FOR NN?
        private float beta1;
        private float beta2;
        private float epsilon;
        public void _Init()
        {
            weights = np.random.randn(input_size, output_size) * np.sqrt(2.0 / input_size); //initialise weights according to HE-Initialisation
            biases = np.zeros(1, output_size);
            m_weights = np.zeros(input_size, output_size);
            v_weights = np.zeros(input_size, output_size);
            m_biases = np.zeros(1, output_size);
            v_biases = np.zeros(1, output_size);
            //initialise hyperparameters for Adam optimiser
            beta1 = 0.9f;
            beta2 = 0.999f;
            epsilon = 1e-8f;
        }
        public NDArray output;
        private NDArray x;
        public NDArray forward(NDArray X) //x is a layer input array to perform the forward pass on
        {
            x = X;

            //calculate the layer output z
            NDArray z = np.dot(x, weights) + biases;

            //apply activation functions to output
            if (activation == "relu")
            {
                output = np.maximum(0, z);
            }
            else if (activation == "softmax")
            {
                NDArray exp_values = np.exp(z - np.max(z, -1, true));
                output = exp_values / np.sum(exp_values, -1, true); //softmax 
            } else if (activation == "tanh"){
                output = np.tanh(z);
            }

            return output;
        }

        private NDArray d_inputs;
        public NDArray backward(NDArray d_values, float learning_rate, int t) //d_values is the derivative of the output, t is the timestep
        {
            var mask = np.zeros(output.shape);
            var zero = np.zeros(1, 1);
            for (int i = 0; i < output.shape[0]; i++)
            {
                for (int j = 0; j < output.shape[1]; j++)
                {
                    int value = output[i, j];
                    if (value > 0) { mask[i, j] = value; }
                }
            }
            //take derivatives of activation function
            if (activation == "relu")
            {
                d_values = d_values * mask;
                int NOFUCKINGWAY = 420;
            }
            else if (activation == "softmax")
            {
                for (int i = 0; i < d_values.shape[0]; i++)
                {
                    NDArray gradient = d_values[i];
                    if (gradient.ndim == 1) //for single instance
                    {
                        gradient = gradient.reshape(-1, 1);
                    }
                    //create jacboian matrix manually because diagflat doesn't exist in numsharp
                    NDArray jacobian_matrix = np.zeros(gradient.shape[0], gradient.shape[0]);
                    for (int j = 0; j < gradient.shape[0]; j++)
                    {
                        jacobian_matrix[i, j] = gradient[j];
                    }

                    jacobian_matrix -= np.dot(gradient, gradient.T);
                    d_values[i] = np.dot(jacobian_matrix, output[i]);
                }
            } else if (activation == "tanh"){
                d_values = 1 - np.power(np.tanh(d_values), 2);
            }

            //calculte derivatives wrt weight and bias
            d_weights = np.dot(x.T, d_values);
            d_biases = FCLayer.SumOverAxis0(d_values);
            //limit derivatives to avoid really big or really small numbers
            d_weights = np.clip(d_weights, -1, 1);
            d_biases = np.clip(d_biases, -1, 1);


            //calculate gradient wrt to input
            d_inputs = np.dot(d_values, weights.T);

            //update weights and biases using learning rate and derivatives
            weights -= learning_rate * d_weights;
            biases -= learning_rate * d_biases;

            /* KILL ADAM */
            //update weights using m and v values (Adam)    
            m_weights = beta1 * m_weights + (1 - beta1) * d_weights;
            v_weights = beta2 * v_weights + (1 - beta2) * np.power(d_weights, 2);
            m_hat_weights = m_weights / (1 - np.power(beta1, t));
            v_hat_weights = v_weights / (1 - np.power(beta2, t));
            weights -= learning_rate * m_hat_weights / (np.sqrt(v_hat_weights) + epsilon);

            //update biases using m and v values (Adam)
            m_biases = beta1 * m_biases + (1 - beta1) * d_biases;
            v_biases = beta2 * v_biases + (1 - beta2) * np.power(d_biases, 2);
            m_hat_biases = m_biases / (1 - np.power(beta1, t));
            v_hat_biases = v_biases / (1 - np.power(beta2, t));
            biases -= learning_rate * m_hat_biases / (np.sqrt(v_hat_biases) + epsilon);
            

            return d_inputs;
        }
        static NDArray SumOverAxis0(NDArray array) //fine... i'll do it myself. implements np.sum(array, axis:0, keepDims = True).
        {
            int rows = array.shape[0];
            int cols = array.shape[1];
            NDArray result = np.zeros(cols);
            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    result[j] += array[i, j];
                }
            }

            return result.reshape(1, cols);
        }
    }

    class CreateModel
    {
        public CreateModel(int Input_Size, int Output_Size, int[] Hidden_Sizes)
        {
            input_size = Input_Size;
            output_size = Output_Size;
            hidden_sizes = Hidden_Sizes;

            _Init();
        }

        public int input_size { get; }
        public int output_size { get; }
        public int[] hidden_sizes { get; }
        public FCLayer layer1 { get; set; }
        public FCLayer layer2 { get; set; }
        public FCLayer layer3 { get; set; }
        private void _Init() //this init will have to be changed if more layers are to be implemented. how to do NEAT? perhaps from scratch, not using this system and letting the program create neurons layers syanpses etc.
        {
            layer1 = new FCLayer(input_size, hidden_sizes[0], "relu");
            layer2 = new FCLayer(hidden_sizes[0], hidden_sizes[1], "relu");
            layer3 = new FCLayer(hidden_sizes[1], output_size, "relu");
        }

        public NDArray forward(NDArray inputs)
        {
            NDArray output1 = layer1.forward(inputs);
            NDArray output2 = layer2.forward(output1);
            NDArray output3 = layer3.forward(output2);
            return output3;
        }

        private int t = 0;
        private float learning_rate;
        private NDArray output_grad;
        private NDArray grad_3;
        private NDArray grad_2;
        private NDArray grad_1;
        public void train(NDArray inputs, NDArray targets, int n_epochs, float initial_learning_rate, float decay)
        {
            //define lists for loss and accuracy
            List<double> loss_log = new List<double>();
            List<float> accuracy_log = new List<float>();
            for (int epoch = 0; epoch < n_epochs; epoch++)
            {
                Console.Write("Epoch " + epoch.ToString() + " of " + n_epochs.ToString());
                //forward pass
                NDArray output = forward(inputs);
                //calculate loss - MSE
                double loss1 = np.mean(np.power((np.subtract(targets, output)), 2));
                //calculate loss - Categorical Crossentropy
                double epsilon = 1e-12;
                double loss2 = np.mean(targets * np.log(output + epsilon));
                loss2 *= -1;
                Console.Write("\n Loss1: " + loss1.ToString() + "\n");
                Console.Write(" Loss2: " + loss2.ToString() + "\n");

                //backwards pass
                //output_grad = 6 / output.shape[0] * (output - targets);
                output = np.clip(output, epsilon, 1 - epsilon);
                output_grad = (- (targets / output) + (1 - targets) / (1 - output)) / output.shape[0];
                Console.WriteLine(output.ToString());
                t++;
                //update learning rate
                learning_rate = initial_learning_rate / (1 + decay * epoch);
                

                if (epoch == 99){
                    Console.WriteLine(layer1.weights.ToString());
                    Console.WriteLine(layer2.weights.ToString());
                    Console.WriteLine(layer3.weights.ToString());
                }
                grad_3 = layer3.backward(output_grad, learning_rate, t);
                grad_2 = layer2.backward(grad_3, learning_rate, t);
                grad_1 = layer1.backward(grad_2, learning_rate, t);
            }
        }
    }


}
