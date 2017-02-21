import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.util.ArrayList;
import java.util.Collections;

import weka.classifiers.Evaluation;
import weka.core.Instance;
import weka.core.Instances;

public class stumpCreator {
	public static Instances concatInstances (Instances inst1, Instances inst2) {
	ArrayList<Instance> instAL = new ArrayList<Instance>();
	for (int i = 0; i < inst2.numInstances(); i++) {
	    instAL.add(inst2.instance(i));
	}
	for (int i = 0; i < instAL.size(); i++) {
	    inst1.add(instAL.get(i));
	}
	return (inst1);
	}
	
	public static Instances getHalfData(Instances data) {
		//Randomly choose instance to include in classification
		int length = data.numInstances();
		Instances halfOfData = new Instances(data);
		int half = length / 2;
		ArrayList<Integer> ifChoose = new ArrayList<Integer>();
		for (int i = 1; i <= half; i++) {
			ifChoose.add(0);
		}
		for (int i = 1; i <= length - half; i++) {
			ifChoose.add(1);
		}
		Collections.shuffle(ifChoose);
		
		for (int i = length - 1; i >= 0; i--) {
			if (ifChoose.get(i) == 0) {
				halfOfData.delete(i);
			}
		} 
		
		return halfOfData;
	}
	
    public static String oneStumpFeature(int foldAsTest) throws Exception {
	
	int DEPTH = 4;
	
	// Load the data
	Instances testData = new Instances(new FileReader(new File("badges.example.arff"))); // Just for initialization, will be overlapped.
	ArrayList<Instances> trainDataList = new ArrayList<Instances>();
	int i = 0;
	for (i = 1; i <= 5; i++) {
		String path = "badges.example.fold"+ i +".arff";
		Instances data = new Instances(new FileReader(new File(path)));
		// The last attribute is the class label
		data.setClassIndex(data.numAttributes() - 1);
		if (i == foldAsTest) {
			testData = data;
		}
		else {
			trainDataList.add(data);
		}
	}

	Instances trainData = trainDataList.get(0);
	for (i = 1; i <= 3; i++ ) {
		trainData = concatInstances(trainData, trainDataList.get(i));
	}
	
	Instances halfOfTrain = getHalfData(trainData);
	Instances test = testData;
	
	
	// Create a new ID3 classifier. This is the modified one where you can
	// set the depth of the tree.
	Id3 classifier = new Id3();

	// An example depth. If this value is -1, then the tree is grown to full
	// depth.
	classifier.setMaxDepth(DEPTH);

	// Train
	classifier.buildClassifier(halfOfTrain);

	// Print the classifier
	//System.out.println(classifier);
	//System.out.println();

	//System.out.println(evaluation.toSummaryString());
	// Output classification result of the stump
	//ArrayList<char> result = new ArrayList<char>();
	String labelList = new String();
	for (i = 0; i < trainData.numInstances(); i++) {
		int label = (int) classifier.classifyInstance(trainData.instance(i));
		
		labelList = labelList.concat(Integer.toString(label));
	}
	labelList = labelList.concat(" ");
	for (i = 0; i < test.numInstances(); i++) {
		int label = (int) classifier.classifyInstance(test.instance(i));
		
		labelList = labelList.concat(Integer.toString(label));
	}
	return labelList;
    }
    
    public static void main(String[] args) throws Exception {
    	int foldAsTest = 0;
    	for (foldAsTest = 1; foldAsTest <=5; foldAsTest ++) {
    		String outputname = "stumpfeatures_fold" + foldAsTest + ".txt";
    		FileWriter fileWriter = new FileWriter(outputname);  
    		for(int i = 0; i < 100; i ++) {
    			String labelList = oneStumpFeature(foldAsTest);
    			fileWriter.write(labelList);
    			fileWriter.write("\n\n");
    		}
    		fileWriter.close();
    	}
    }
    	
}
