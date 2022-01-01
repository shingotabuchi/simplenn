using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
public class NeuralNet : MonoBehaviour
{
    float[,] inputs = new float[6,3]{
        {0,0,1},
        {0,1,1},
        {1,0,0},
        {1,1,0},
        {1,0,1},
        {1,1,1},
    };
    float[] desiredOutputs = new float[6]{
       0,
       1,
       0,
       1,
       1,
       0,
    }; 
    float[,] hiddenWeights = new float[3,4];
    float[] outputWeights = new float[4];
    float[] hiddenLayerOutputs = new float[3];
    float outputLayerOutput;
    public int iterationCount;
    public float alpha;
    public GameObject[] TableRowCanvas = new GameObject[6]; 
    public GameObject[] InputNodesCanvas = new GameObject[3];
    public GameObject[] HiddenNodesCanvas = new GameObject[3];
    public GameObject OutputNodeCanvas;
    Text[] yosokuTexts = new Text[6];
    Text[] inputNodeOutputs = new Text[3];
    Text[] hiddenNodeOutputs = new Text[3];
    Text outputNodeOutput;
    Text[,] hiddenWeightText = new Text[3,3];
    GameObject[,] hiddenWeightLine = new GameObject[3,3];
    Text[] outputWeightText = new Text[3];
    GameObject[] outputWeightLine =  new GameObject[3];
    public int selectedDataIndex = 0;
    float lineScale;
    public float C;
    GameObject[] highlight = new GameObject[6];
    void Start(){
        for (int i = 0; i < 4; i++)
        {
            outputWeights[i] = Random.Range(-1f,1f);
            for (int j = 0; j < 3; j++)
            {
                hiddenWeights[j,i] = Random.Range(-1f,1f);
            }
        }
        for (int i = 0; i < 6; i++)
        {
            for (int j = 0; j < 3; j++)
            {
                Transform nyuuryoku = TableRowCanvas[i].transform.GetChild(j);
                Text nyuuryokuText = nyuuryoku.gameObject.GetComponent<Text>();
                nyuuryokuText.text = inputs[i,j].ToString();
            }
            Transform nyuuryokuYosoku = TableRowCanvas[i].transform.GetChild(3);
            yosokuTexts[i] = nyuuryokuYosoku.gameObject.GetComponent<Text>();
            Transform seikai = TableRowCanvas[i].transform.GetChild(4);
            highlight[i] = TableRowCanvas[i].transform.GetChild(5).gameObject;
            Text seikaiText = seikai.gameObject.GetComponent<Text>();
            seikaiText.text = desiredOutputs[i].ToString();
        }
        for (int i = 0; i < 3; i++)
        {
            inputNodeOutputs[i] = InputNodesCanvas[i].transform.GetChild(0).gameObject.GetComponent<Text>();
            hiddenNodeOutputs[i] = HiddenNodesCanvas[i].transform.GetChild(0).gameObject.GetComponent<Text>();
            outputWeightText[i] = OutputNodeCanvas.transform.GetChild(1+i).gameObject.GetComponent<Text>();
            outputWeightLine[i] = OutputNodeCanvas.transform.GetChild(1+i).GetChild(0).gameObject;
            lineScale = outputWeightLine[i].transform.localScale.y;
            for (int j = 0; j < 3; j++)
            {
                hiddenWeightText[i,j] = HiddenNodesCanvas[i].transform.GetChild(1+j).gameObject.GetComponent<Text>();
                hiddenWeightLine[i,j] = HiddenNodesCanvas[i].transform.GetChild(1+j).GetChild(0).gameObject;
            }
        }
        outputNodeOutput = OutputNodeCanvas.transform.GetChild(0).gameObject.GetComponent<Text>();
    }
    float Sigmoid(float x){
        return 1f / (1f + Mathf.Exp(-x));
    }
    float dSigmoid(float x){
        return Mathf.Exp(-x) / ((1f + Mathf.Exp(-x))*(1f + Mathf.Exp(-x)));
    }
    void Update(){
        float[,] total_hidden_gradient = new float[3,4];
        float[] total_output_gradient = new float[4];
        for (int j = 0; j < 4; j++)
        {
            total_output_gradient[j] = 0;
            for (int k = 0; k < 3; k++)
            {
                total_hidden_gradient[k,j] = 0;
            }
        }
        for (int j = 0; j < 6; j++)
        {   
            outputLayerOutput = outputWeights[3];
            for (int k = 0; k < 3; k++)
            {
                hiddenLayerOutputs[k] = hiddenWeights[k,3];
                for (int l = 0; l < 3; l++)
                {
                    hiddenLayerOutputs[k] += hiddenWeights[k,l] * inputs[j,l];
                }
                hiddenLayerOutputs[k] = Sigmoid(hiddenLayerOutputs[k]);
                outputLayerOutput += hiddenLayerOutputs[k] * outputWeights[k];
            }
            yosokuTexts[j].text = outputLayerOutput.ToString();
            float outputErr = outputLayerOutput - desiredOutputs[j];
            float[] hiddenErr = new float[3];
            for (int k = 0; k < 3; k++)
            {
                hiddenErr[k] = hiddenLayerOutputs[k]*(1-hiddenLayerOutputs[k])*outputErr*outputWeights[k];
            }
            for (int k = 0; k < 4; k++)
            {
                if(k!=3)total_output_gradient[k] += hiddenLayerOutputs[k]*outputErr/6f;
                else total_output_gradient[k] += 1*outputErr/6f;
                for (int l = 0; l < 3; l++)
                {
                    if(k!=3) total_hidden_gradient[l,k] += inputs[j,k]*hiddenErr[l]/6f;
                    else total_hidden_gradient[l,k] += 1*hiddenErr[l]/6f;
                }
            }

            if(j==selectedDataIndex){
                for (int k = 0; k < 3; k++)
                {
                    hiddenNodeOutputs[k].text = hiddenLayerOutputs[k].ToString();
                    outputWeightText[k].text = outputWeights[k].ToString();
                    inputNodeOutputs[k].text = inputs[j,k].ToString();
                    outputWeightLine[k].transform.localScale = new Vector3(outputWeightLine[k].transform.localScale.x,lineScale * Mathf.Log(1+C*Mathf.Abs(outputWeights[k])),outputWeightLine[k].transform.localScale.z);
                    SpriteRenderer renderer = outputWeightLine[k].GetComponent<SpriteRenderer>();
                    float W = 0.5f*(1-outputWeights[k]);
                    if(W>1)W=1;
                    if(W<0)W=0;
                    renderer.color = new Color(W,W,W,1);
                    for (int l = 0; l < 3; l++)
                    {
                        hiddenWeightText[k,l].text = hiddenWeights[k,l].ToString();
                        hiddenWeightLine[k,l].transform.localScale = new Vector3(hiddenWeightLine[k,l].transform.localScale.x,lineScale * Mathf.Log(1+C*Mathf.Abs(hiddenWeights[k,l])),hiddenWeightLine[k,l].transform.localScale.z);
                        renderer = hiddenWeightLine[k,l].GetComponent<SpriteRenderer>();
                        W = 0.5f*(1-hiddenWeights[k,l]);
                        if(W>1)W=1;
                        if(W<0)W=0;
                        renderer.color = new Color(W,W,W,1);
                    }
                }
                outputNodeOutput.text = yosokuTexts[j].text;
            }
        }
        for (int j = 0; j < 4; j++)
        {
            outputWeights[j] -= alpha*total_output_gradient[j];
            for (int k = 0; k < 3; k++)
            {
                hiddenWeights[k,j] -= alpha*total_hidden_gradient[k,j];
            }
        }
    }
    public void UpSelectedData(){
        selectedDataIndex--;
        if(selectedDataIndex<0) selectedDataIndex = 0;
        UpdateHighlight();
    }   
    public void LowSelectedData(){
        selectedDataIndex++;
        if(selectedDataIndex>5) selectedDataIndex = 5;
        UpdateHighlight();
    }   
    public void UpdateHighlight(){
        for (int i = 0; i < 6; i++)
        {
            if(i==selectedDataIndex)highlight[i].SetActive(true);
            else highlight[i].SetActive(false);
        }
    }
}



// import numpy as np

// # define the sigmoid function
// def sigmoid(x, derivative=False):

//     if (derivative == True):
//         return sigmoid(x,derivative=False) * (1 - sigmoid(x,derivative=False))
//     else:
//         return 1 / (1 + np.exp(-x))

// # choose a random seed for reproducible results
// np.random.seed(1)

// # learning rate
// alpha = .1

// # number of nodes in the hidden layer
// num_hidden = 3

// # inputs
// X = np.array([  
//     [0, 0, 1],
//     [0, 1, 1],
//     [1, 0, 0],
//     [1, 1, 0],
//     [1, 0, 1],
//     [1, 1, 1],
// ])

// # outputs
// # x.T is the transpose of x, making this a column vector
// y = np.array([[0, 1, 0, 1, 1, 0]]).T

// # initialize weights randomly with mean 0 and range [-1, 1]
// # the +1 in the 1st dimension of the weight matrices is for the bias weight
// hidden_weights = 2*np.random.random((X.shape[1] + 1, num_hidden)) - 1
// output_weights = 2*np.random.random((num_hidden + 1, y.shape[1])) - 1

// # number of iterations of gradient descent
// num_iterations = 10000

// # for each iteration of gradient descent
// for i in range(num_iterations):

//     # forward phase
//     # np.hstack((np.ones(...), X) adds a fixed input of 1 for the bias weight
//     input_layer_outputs = np.hstack((np.ones((X.shape[0], 1)), X))
//     hidden_layer_outputs = np.hstack((np.ones((X.shape[0], 1)), sigmoid(np.dot(input_layer_outputs, hidden_weights))))
//     output_layer_outputs = np.dot(hidden_layer_outputs, output_weights)

//     # backward phase
//     # output layer error term
//     output_error = output_layer_outputs - y
//     # hidden layer error term
//     # [:, 1:] removes the bias term from the backpropagation
//     hidden_error = hidden_layer_outputs[:, 1:] * (1 - hidden_layer_outputs[:, 1:]) * np.dot(output_error, output_weights.T[:, 1:])

//     # partial derivatives
//     hidden_pd = input_layer_outputs[:, :, np.newaxis] * hidden_error[: , np.newaxis, :]
//     output_pd = hidden_layer_outputs[:, :, np.newaxis] * output_error[:, np.newaxis, :]

//     # average for total gradients
//     total_hidden_gradient = np.average(hidden_pd, axis=0)
//     total_output_gradient = np.average(output_pd, axis=0)

//     # update weights
//     hidden_weights += - alpha * total_hidden_gradient
//     output_weights += - alpha * total_output_gradient

// # print the final outputs of the neural network on the inputs X
// print("Output After Training: \n{}".format(output_layer_outputs))