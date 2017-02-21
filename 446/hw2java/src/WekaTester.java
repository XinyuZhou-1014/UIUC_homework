import java.io.File;
import java.io.FileReader;
import java.util.ArrayList;
import java.lang.String;

import weka.classifiers.Evaluation;
import weka.core.Instance;
import weka.core.Instances;

public class WekaTester {
	
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

    public static void run(int foldAsTest, int DEPTH, boolean silent) throws Exception {

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
	
	Instances train = trainData;
	Instances test = testData;

	// Create a new ID3 classifier. This is the modified one where you can
	// set the depth of the tree.
	Id3 classifier = new Id3();

	// An example depth. If this value is -1, then the tree is grown to full
	// depth.
	classifier.setMaxDepth(DEPTH);

	// Train
	classifier.buildClassifier(train);

	// Print the classifier
	if (silent == false){
	    System.out.println(classifier);
	}
	System.out.println();

	// Evaluate on the test set
	Evaluation evaluation = new Evaluation(test);
	evaluation.evaluateModel(classifier, test);
	System.out.println(evaluation.toSummaryString());

    }
    
    public static void main(String args[]) throws Exception {
    	if (args.length != 1) {
    	    System.err.println("Usage: WekaTester arff-file");
    	    System.exit(-1);
    	}
    	int depth = Integer.parseInt(args[0]);
    	for(int i=1;i<=5;i++){
    	    run(i, depth, false);
    	}
    }
}
