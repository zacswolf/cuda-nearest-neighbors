#include "nearest_neighbor_sequential.h"
#include "vector"

template <typename T, typename G>
G* seqNormal(Matrix<T> &trainData, Matrix<G> &trainLabels, Matrix<T> &testData) {
	printf("seqNormal\n");
	int numPredictPoints = testData.numRows;
	int numDataPoints = trainData.numRows;

	printf("A\n");
	G *predictedLabels = new G(numPredictPoints);
	printf("B\n");
	// Nearest neighbors
	for (int currentTestPoint = 0; currentTestPoint < numPredictPoints; currentTestPoint++) {
		int closestPoint = 0;
		double closestDistance = std::numeric_limits<double>::max();
		for (int currentTrainPoint = 0; currentTrainPoint < numDataPoints; currentTrainPoint++) {
			// l2 distance squared
			double currentDistance = Matrix<T>::l2RowDistanceSeq(trainData, currentTrainPoint, testData, currentTestPoint);

			// Save if currentDistance < closestDistance
			bool newClosest = (currentDistance < closestDistance);

			closestPoint = (!newClosest)*closestPoint + newClosest*currentTrainPoint;
			closestDistance = (!newClosest)*closestDistance + newClosest*currentDistance;
		}
		predictedLabels[currentTestPoint] = trainLabels.data[closestPoint];
	}

	return predictedLabels;
}

template <typename T, typename G>
G* seqJLGaussian(Matrix<T> &trainData, Matrix<G> &trainLabels, Matrix<T> &testData, int newDim) {
	int dim = trainData.numCols;

	// Make a random projection matrix of size dim x newDim
	Matrix<float> rpMat = Matrix<float>(dim, newDim);
	std::normal_distribution<float> distribution(0., 1.);
	rpMat.fill(distribution);

	// newTrainData = trainData x rpMat, numDataPoints by newDim
	Matrix<decltype(std::declval<T&>() * std::declval<G&>())> newTrainData = Matrix<T>::matMulSeq(trainData, rpMat);

	// newTestData = testData x rpMat, numDataPoints by newDim
	Matrix<decltype(std::declval<T&>() * std::declval<G&>())> newTestData = Matrix<T>::matMulSeq(testData, rpMat);

	return seqNormal(newTrainData, trainLabels, newTestData);
}

template <typename T, typename G>
G* seqJLBernoulli(Matrix<T> &trainData, Matrix<G> &trainLabels, Matrix<T> &testData, int newDim) {
	int dim = trainData.numCols;

	// Make a random projection matrix of size dim x newDim
	Matrix<int8_t> rpMat = Matrix<int8_t>(dim, newDim);
	std::bernoulli_distribution distribution(.5);
	rpMat.fill(distribution);

	// newTrainData = trainData x rpMat, numDataPoints by newDim

	Matrix<decltype(std::declval<T&>() * std::declval<G&>())> newTrainData = Matrix<T>::matMulSeq(trainData, rpMat);

	// newTestData = testData x rpMat, numDataPoints by newDim
	Matrix<decltype(std::declval<T&>() * std::declval<G&>())> newTestData = Matrix<T>::matMulSeq(testData, rpMat);

	return seqNormal(newTrainData, trainLabels, newTestData);
}

template <typename T>
Matrix<T> applySHD(Matrix<T> &dataT, int newDim, Matrix<int8_t> &D, Matrix<int> &S) {
	printf("\n\n+++++applySHD++++++\n");
	int dim = dataT.numRows;
	int numPoints = dataT.numCols;
	printf("numPoints %i\n",numPoints);

	assert(dim==D.numRows);
	assert(newDim==S.numRows);

	printf("======D\n");
	D.print();

	// Mult D x trainDataT[pointIdx]
	Matrix<T> result1 = Matrix<T>(dim, numPoints);	

	for (int pointIdx=0; pointIdx < numPoints; pointIdx++) {
		for (int d=0; d < dim; d++) {
			result1.data[result1.index(d, pointIdx)] = D.data[d] * dataT.data[dataT.index(d, pointIdx)];
		}
	}


	printf("======result1\n");
	result1.print();

	// Mult H x result1 
	Matrix<T> result2 = Matrix<T>(dim, numPoints);

	assert(dim>1);
	
	int log2dim = ceil(log2(dim));
	int hShape = pow(2,log2dim);

	
	for (int pointIdx=0; pointIdx < numPoints; pointIdx++) {
		
		// 2x2 matrix
		

		int order = 1;
		int stride = 2;
		int split = stride/2;

		Matrix<T> mats [] = {Matrix<T>(hShape, 1), Matrix<T>(hShape, 1)};
		mats[0].fill(0);
		mats[1].fill(0);
		printf("======AAAAAAA\n");

		int newIdx = 0;

		for (int i = 0; i < dim; i++) {
			mats[newIdx].data[i] = result1.data[result1.index(i, pointIdx)];
		}
		printf("======newMat\n");
		mats[newIdx].print();

		// temp = last[:split];
		// last[:split] += last[split:shape];
		// last[split:shape] += temp;


		for (order = 2; order < log2dim; order++) { // sequential loop
			newIdx = !newIdx;

			stride = pow(2, order);
			split = stride/2;

			// delete lastMat
			// delete [] lastMat->data;
			// delete lastMat;

			// shift new to old
			// lastMat = newMat;
			// newMat = new Matrix<T>(hShape, 1);

			for (int strideId = 0; strideId < hShape/stride; strideId++) { // could parallize
				for (int idx = 0; idx < split; idx++) {
					// c0
					mats[newIdx].data[strideId*stride+idx] = mats[!newIdx].data[strideId*stride+idx] + mats[!newIdx].data[strideId*stride+idx+(split/2)];

					// c1
					mats[newIdx].data[strideId*stride+idx+split] = mats[!newIdx].data[strideId*stride+idx+split] - mats[!newIdx].data[strideId*stride+idx+(split/2)+split];
				}
			}

		}

		printf("======newMat\n");
		mats[newIdx].print();

		for (int d=0; d<dim; d++) {
			result2.data[result2.index(d, pointIdx)] = mats[newIdx].data[d];
		}

		//CLEANUP
		delete [] mats[0].data;
		delete [] mats[1].data;
	} 

	printf("======result2\n");
	result2.print();

	// Mult S x result2 
	Matrix<T> result3 = Matrix<T>(newDim, numPoints);

	printf("======S\n");
	S.print();
	


	for (int pointIdx=0; pointIdx < numPoints; pointIdx++) {
		for(int d = 0; d < S.numRows; d++) {
			int onehotdim = S.data[d];
			result3.data[result3.index(d, pointIdx)] = result2.data[result2.index(onehotdim, pointIdx)];
		}
	}

	printf("======result3\n");
	result3.print();


	//CLEANUP
	delete [] result1.data;
	delete [] result2.data;


	return result3;
}

template <typename T, typename G>
G* seqJLFast(Matrix<T> &trainData, Matrix<G> &trainLabels, Matrix<T> &testData, int newDim) {
	printf("seqJLFast, trainData %d %d, trainLabels %d %d, testData %d %d,\n", trainData.numRows, trainData.numCols, trainLabels.numRows, trainLabels.numCols, testData.numRows, testData.numCols);


	assert(trainData.numCols == testData.numCols);
	int dim = trainData.numCols;

	// SHD is a newDim x dim matrix

	// scalar for SHD
	float scalar = sqrt(static_cast<float>(dim)/newDim);


	// make S, a vector of indices to represent the rows of a matrix with one-hot rows
	// Matrix<int> S = Matrix<int>(newDim, dim);
	Matrix<int> S = Matrix<int>(newDim, 1);

	printf("dim %i\n", dim);
	std::uniform_int_distribution<> distribution(0, dim-1);
	std::default_random_engine generator(0);

	for (int rowIdx = 0; rowIdx < S.numRows; rowIdx++) {
		S.data[S.index(rowIdx,0)] = distribution(generator);
	}

	// S.fill(0);
	printf("S.numRows %d\n", S.numRows);
	for (int rowIdx = 0; rowIdx < S.numRows; rowIdx++) {
		S.data[S.index(rowIdx, distribution(generator))] = 1;
	}
	S.print();

	
	// make H, Hadamard matrix but could be Fourier matrix
	// we use Sylvester’s Construction to make the Hadamard matrix

	
	/*
	printf("dim %d, log2dim %d, hShape %d\n", dim, log2dim, hShape);
	Matrix<int8_t> H = Matrix<int8_t>(hShape, hShape);

	H.data[H.index(0,0)] = 1;


	assert(hShape>0);

	for (int order = 0; order < log2dim; order++) {
		// copy rows 0:2^order and columns 0:2^order to destinations
		int shape = pow(2, order);
		for (int row=0; row < shape; row++) {
			for (int col=0; col < shape; col++) {
				int val = H.data[H.index(row, col)];
				// left
				H.data[H.index(row, col+shape)] = val;
				
				// bottom
				H.data[H.index(row+shape, col)] = val;

				// diag
				H.data[H.index(row+shape, col+shape)] = -1*val;
			}
		}
	}

	#ifdef TESTMATH
		printf("H matrix\n");
		H.print();

		Matrix<int8_t> HT = H.transpose();
		Matrix<int8_t> ident = Matrix<int8_t>::matMulSeq(HT, H);
		printf("identity times scalar matrix\n");
		ident.print();
	#endif
	*/

	// make, D a Rademacher vector (unif +- 1) representing a diagonal matrix
	Matrix<int8_t> D = Matrix<int8_t>(dim, 1);
	std::bernoulli_distribution distribution2(.5);
	D.fill(distribution2);


	// SHDx runs in dlog(d)

	Matrix<T> trainDataT = trainData.transpose();
	Matrix<T> testDataT = testData.transpose();

	printf("seqJLFast, trainDataT %d %d, trainLabels %d %d, testDataT %d %d,\n", trainDataT.numRows, trainDataT.numCols, trainLabels.numRows, trainLabels.numCols, testDataT.numRows, testDataT.numCols);
	Matrix<T> newTrainDataT = applySHD(trainDataT, newDim, D, S);
	Matrix<T> newTestDataT = applySHD(testDataT, newDim, D, S);

	Matrix<T> newTrainData = newTrainDataT.transpose();
	Matrix<T> newTestData = newTestDataT.transpose();
	
	printf("newTrainData\n");
	newTrainData.print();

	printf("newTestData\n");
	newTestData.print();

	//CLEANUP
	delete [] trainDataT.data;
	delete [] testDataT.data;
	delete [] newTrainDataT.data;
	delete [] newTestDataT.data;
	delete [] S.data;
	delete [] D.data;


	printf("seqNormal, newTrainData %d %d, trainLabels %d %d, newTestData %d %d,\n", newTrainData.numRows, newTrainData.numCols, trainLabels.numRows, trainLabels.numCols, newTestData.numRows, newTestData.numCols);
	
	return seqNormal(newTrainData, trainLabels, newTestData);
}
