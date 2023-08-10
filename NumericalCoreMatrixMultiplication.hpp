
// Author: Jakub Lisowski

// Collection of possible solutions of the matrix multiplication problem made by me for educational purposes,
// each of them allowed me better understand how optimisation of numerical calculations should be done


#ifndef PARALLELNUM_MATRIX_MULTIPLICATION_H_
#define PARALLELNUM_MATRIX_MULTIPLICATION_H_

#include <immintrin.h>
#include <cmath>
#include <iostream>
#include <cstdlib>

template<typename T>
class SimpleMultMachine {
	const unsigned Src1Rows;
	const unsigned Src1Cols;
	const unsigned Src2Rows;
	const unsigned Src2Cols;
	const unsigned TargetSizeOfLine;
	const unsigned Src1SizeOfLine;
	const unsigned Src2SizeOfLine;
	T* const Target;
	T* const Src1;
	T* const Src2;
	static const unsigned ElementsInCacheLine = CACHE_LINE / sizeof(T);
public:
	SimpleMultMachine(const unsigned Src1Rows,
		const unsigned Src1Cols,
		const unsigned Src2Rows,
		const unsigned Src2Cols,
		const unsigned TargetSizeOfLine,
		const unsigned Src1SizeOfLine,
		const unsigned Src2SizeOfLine,
		T* Target,
		T* const Src1,
		T* const Src2) :
		Src1Rows{ Src1Rows }, Src1Cols{ Src1Cols }, Src2Rows{ Src2Rows }, Src2Cols{ Src2Cols },
		TargetSizeOfLine{ TargetSizeOfLine }, Src1SizeOfLine{ Src1SizeOfLine }, Src2SizeOfLine{ Src2SizeOfLine },
		Target{ Target }, Src1{ Src1 }, Src2{ Src2 }
	{}

	void MultAlgo()
		// 0.33
	{
		for (unsigned i = 0; i < Src2Cols; ++i) {
			for (unsigned j = 0; j < Src2Rows; ++j) {
				T Val = Src2[i * Src2SizeOfLine + j];
				for (unsigned k = 0; k < Src1Rows; ++k) {
					Target[i * TargetSizeOfLine + k] += Src1[j * Src1SizeOfLine + k] * Val;
				}
			}
		}
	}

	void MultAlgo1_CC()
		// Most common matrix multipliacation algorithm with blocking aligned to cache line
		// Actualy works ONLY on perfectly blockabled matrices e.g nxn, where n is divisible by ElementsInCacheLine
		// 0.23
	{
		const unsigned BlocksPerRun = (Src1Cols / ElementsInCacheLine) * ElementsInCacheLine;
		const unsigned TargetVerticalBlocks = (Src1Rows / ElementsInCacheLine) * ElementsInCacheLine;
		const unsigned TargetHorizontalBlocks = (Src2Cols / ElementsInCacheLine) * ElementsInCacheLine;

		for (unsigned i = 0; i < TargetHorizontalBlocks; i += ElementsInCacheLine) {
			for (unsigned j = 0; j < TargetVerticalBlocks; j += ElementsInCacheLine) {
#define SumOverVectors(vector_offset) Src1[(k + kk) * Src1SizeOfLine + j + jj] * Src2[(i + ii + vector_offset) * Src2SizeOfLine + k + kk];
#define SaveAccumulatorsOverVector(vector_offset) Target[(i + ii + vector_offset) * TargetSizeOfLine + j + jj]

				for (unsigned k = 0; k < BlocksPerRun; k += ElementsInCacheLine) {
					for (unsigned ii = 0; ii < ElementsInCacheLine; ii += 4) {
						for (unsigned jj = 0; jj < ElementsInCacheLine; ++jj) {
							T acc[4] = { 0 };
							for (unsigned kk = 0; kk < ElementsInCacheLine; ++kk) {
								acc[0] += SumOverVectors(0);
								acc[1] += SumOverVectors(1);
								acc[2] += SumOverVectors(2);
								acc[3] += SumOverVectors(3);
							}

							SaveAccumulatorsOverVector(0) += acc[0];
							SaveAccumulatorsOverVector(1) += acc[1];
							SaveAccumulatorsOverVector(2) += acc[2];
							SaveAccumulatorsOverVector(3) += acc[3];
						}
					}
				}
			}
		}
	}


	// Global assumption used in descritors
	// XYVectorScalar - X and Y can be replaced with B - Blocked or N - Not blocked, which indicates if the descriptors 
	// is meant to be used with whole cache lines or only parts of them. X refers to horizontal cache lines and Y,
	// predictably, to vertical ones

#define BBScaledVectorCoefPacked(offset) Src1[(j + jj) * Src1SizeOfLine + k + kk + offset] * VectorScalar
	// Scaled vector coef, used to get multiple scaled coefs with signle val at once(4)
#define BBSaveAccumulatedCoefsToTarget(offset) Target[(i + ii) * TargetSizeOfLine + k + kk]
	// After accumulating desired amount of vectors coef accumulator variable is added to
	// proper target storage

#define NBScaledVectorCoefPacked(offset) acc0 += Src1[j * Src1SizeOfLine + k + kk + offset] * VectorScalar

	void MultAlgo2_CC_Blocks_EE(const unsigned VectorsBlocks, const unsigned BlocksPerVector, const unsigned BlocksPerBaseVectors,
		const unsigned VectorsStartingBlock = 0)
	{
		for (unsigned i = VectorsStartingBlock; i < VectorsBlocks; i += ElementsInCacheLine) {
			for (unsigned j = 0; j < BlocksPerVector; j += ElementsInCacheLine) {
				for (unsigned k = 0; k < BlocksPerBaseVectors; k += ElementsInCacheLine) {
					for (unsigned ii = 0; ii < ElementsInCacheLine; ++ii) {
						for (unsigned kk = 0; kk < ElementsInCacheLine; kk += 4) {
							T acc0 = 0;
							T acc1 = 0;
							T acc2 = 0;
							T acc3 = 0;

							for (unsigned jj = 0; jj < ElementsInCacheLine; ++jj) {
								T VectorScalar = Src2[(i + ii) * Src2SizeOfLine + j + jj];

								acc0 += BBScaledVectorCoefPacked(0);
								acc1 += BBScaledVectorCoefPacked(1);
								acc2 += BBScaledVectorCoefPacked(2);
								acc3 += BBScaledVectorCoefPacked(3);
							}


							BBSaveAccumulatedCoefsToTarget(0) += acc0;
							BBSaveAccumulatedCoefsToTarget(1) += acc1;
							BBSaveAccumulatedCoefsToTarget(2) += acc2;
							BBSaveAccumulatedCoefsToTarget(3) += acc3;

						}
					}
				}
			}
		}
	}

	void MultAlgo2_CC_Blocks_EN(const unsigned VectorsBlocks, const unsigned BlocksPerVector, const unsigned BlocksPerBaseVectors,
		const unsigned VectorsStartingBlock = 0) {
		for (unsigned i = VectorsStartingBlock; i < VectorsBlocks; i += ElementsInCacheLine) {
			for (unsigned j = 0; j < BlocksPerVector; j += ElementsInCacheLine) {
				for (unsigned k = 0; k < BlocksPerBaseVectors; k += ElementsInCacheLine) {
					for (unsigned ii = 0; ii < ElementsInCacheLine; ++ii) {
						for (unsigned kk = 0; kk < ElementsInCacheLine; kk += 4) {
							T acc0 = 0;
							T acc1 = 0;
							T acc2 = 0;
							T acc3 = 0;

							for (unsigned jj = 0; jj < ElementsInCacheLine; ++jj) {
								T VectorScalar = Src2[(i + ii) * Src2SizeOfLine + j + jj];

								acc0 += BBScaledVectorCoefPacked(0);
								acc1 += BBScaledVectorCoefPacked(1);
								acc2 += BBScaledVectorCoefPacked(2);
								acc3 += BBScaledVectorCoefPacked(3);
							}

							BBSaveAccumulatedCoefsToTarget(0) += acc0;
							BBSaveAccumulatedCoefsToTarget(1) += acc1;
							BBSaveAccumulatedCoefsToTarget(2) += acc2;
							BBSaveAccumulatedCoefsToTarget(3) += acc3;

						}
					}
				}
			}

			for (unsigned k = 0; k < BlocksPerBaseVectors; k += ElementsInCacheLine)
				// Doing last run on the not Blocked Base Vectors
			{
				for (unsigned ii = 0; ii < ElementsInCacheLine; ++ii) {

					for (unsigned kk = 0; kk < ElementsInCacheLine; kk += 4) {
						T acc0 = 0;
						T acc1 = 0;
						T acc2 = 0;
						T acc3 = 0;

						for (unsigned j = BlocksPerVector; j < Src1Cols; ++j) {
							T VectorScalar = Src2[(i + ii) * Src2SizeOfLine + j];

							acc0 += NBScaledVectorCoefPacked(0);
							acc1 += NBScaledVectorCoefPacked(1);
							acc2 += NBScaledVectorCoefPacked(2);
							acc3 += NBScaledVectorCoefPacked(3);
						}


						BBSaveAccumulatedCoefsToTarget(0) += acc0;
						BBSaveAccumulatedCoefsToTarget(1) += acc1;
						BBSaveAccumulatedCoefsToTarget(2) += acc2;
						BBSaveAccumulatedCoefsToTarget(3) += acc3;

					}
				}
			}
		}
	}

	void MultAlgo2_CC_Blocks_NE(const unsigned VectorsBlocks, const unsigned BlocksPerVector, const unsigned BlocksPerBaseVectors,
		const unsigned VectorsStartingBlock = 0) {
		for (unsigned i = VectorsStartingBlock; i < VectorsBlocks; i += ElementsInCacheLine) {
			for (unsigned j = 0; j < BlocksPerVector; j += ElementsInCacheLine) {
				for (unsigned k = 0; k < BlocksPerBaseVectors; k += ElementsInCacheLine) {
					for (unsigned ii = 0; ii < ElementsInCacheLine; ++ii) {
						for (unsigned kk = 0; kk < ElementsInCacheLine; kk += 4) {
							T acc0 = 0;
							T acc1 = 0;
							T acc2 = 0;
							T acc3 = 0;

							for (unsigned jj = 0; jj < ElementsInCacheLine; ++jj) {
								T Val = Src2[(i + ii) * Src2SizeOfLine + j + jj];

								acc0 += Src1[(j + jj) * Src1SizeOfLine + k + kk] * Val;
								acc1 += Src1[(j + jj) * Src1SizeOfLine + k + kk + 1] * Val;
								acc2 += Src1[(j + jj) * Src1SizeOfLine + k + kk + 2] * Val;
								acc3 += Src1[(j + jj) * Src1SizeOfLine + k + kk + 3] * Val;
							}


							Target[(i + ii) * TargetSizeOfLine + k + kk] += acc0;
							Target[(i + ii) * TargetSizeOfLine + k + kk + 1] += acc1;
							Target[(i + ii) * TargetSizeOfLine + k + kk + 2] += acc2;
							Target[(i + ii) * TargetSizeOfLine + k + kk + 3] += acc3;

						}
					}
				}

				// Not guaranteed to be divisible by 4 - switching to alternative algorithm
				for (unsigned ii = 0; ii < ElementsInCacheLine; ii += 4) {
					for (unsigned k = BlocksPerBaseVectors; k < Src1Rows; ++k) {
						T acc0 = 0;
						T acc1 = 0;
						T acc2 = 0;
						T acc3 = 0;

						for (unsigned jj = 0; jj < ElementsInCacheLine; ++jj) {
							acc0 += Src1[(j + jj) * Src1SizeOfLine + k] * Src2[(i + ii) * Src2SizeOfLine + j + jj];
							acc1 += Src1[(j + jj) * Src1SizeOfLine + k] * Src2[(i + ii + 1) * Src2SizeOfLine + j + jj];
							acc2 += Src1[(j + jj) * Src1SizeOfLine + k] * Src2[(i + ii + 2) * Src2SizeOfLine + j + jj];
							acc3 += Src1[(j + jj) * Src1SizeOfLine + k] * Src2[(i + ii + 3) * Src2SizeOfLine + j + jj];
						}


						Target[(i + ii) * TargetSizeOfLine + k] += acc0;
						Target[(i + ii + 1) * TargetSizeOfLine + k] += acc1;
						Target[(i + ii + 2) * TargetSizeOfLine + k] += acc2;
						Target[(i + ii + 3) * TargetSizeOfLine + k] += acc3;

					}
				}
			}
		}
	}

	void MultAlgo2_CC_Blocks_NN(const unsigned VectorsBlocks, const unsigned BlocksPerVector, const unsigned BlocksPerBaseVectors,
		const unsigned VectorsStartingBlock = 0) {
		for (unsigned i = VectorsStartingBlock; i < VectorsBlocks; i += ElementsInCacheLine) {
			for (unsigned j = 0; j < BlocksPerVector; j += ElementsInCacheLine) {
				for (unsigned k = 0; k < BlocksPerBaseVectors; k += ElementsInCacheLine) {
					for (unsigned ii = 0; ii < ElementsInCacheLine; ++ii) {
						for (unsigned kk = 0; kk < ElementsInCacheLine; kk += 4) {
							T acc0 = 0;
							T acc1 = 0;
							T acc2 = 0;
							T acc3 = 0;

							for (unsigned jj = 0; jj < ElementsInCacheLine; ++jj) {
								T Val = Src2[(i + ii) * Src2SizeOfLine + j + jj];

								acc0 += Src1[(j + jj) * Src1SizeOfLine + k + kk] * Val;
								acc1 += Src1[(j + jj) * Src1SizeOfLine + k + kk + 1] * Val;
								acc2 += Src1[(j + jj) * Src1SizeOfLine + k + kk + 2] * Val;
								acc3 += Src1[(j + jj) * Src1SizeOfLine + k + kk + 3] * Val;
							}


							Target[(i + ii) * TargetSizeOfLine + k + kk] += acc0;
							Target[(i + ii) * TargetSizeOfLine + k + kk + 1] += acc1;
							Target[(i + ii) * TargetSizeOfLine + k + kk + 2] += acc2;
							Target[(i + ii) * TargetSizeOfLine + k + kk + 3] += acc3;

						}
					}
				}

				// Not guaranteed to be divisible by 4 - switching to alternative algorithm
				for (unsigned ii = 0; ii < ElementsInCacheLine; ii += 4) {
					for (unsigned k = BlocksPerBaseVectors; k < Src1Rows; ++k) {
						T acc0 = 0;
						T acc1 = 0;
						T acc2 = 0;
						T acc3 = 0;

						for (unsigned jj = 0; jj < ElementsInCacheLine; ++jj) {
							T Val = Src1[(j + jj) * Src1SizeOfLine + k];

							acc0 += Val * Src2[(i + ii) * Src2SizeOfLine + j + jj];
							acc1 += Val * Src2[(i + ii + 1) * Src2SizeOfLine + j + jj];
							acc2 += Val * Src2[(i + ii + 2) * Src2SizeOfLine + j + jj];
							acc3 += Val * Src2[(i + ii + 3) * Src2SizeOfLine + j + jj];
						}


						Target[(i + ii) * TargetSizeOfLine + k] += acc0;
						Target[(i + ii + 1) * TargetSizeOfLine + k] += acc1;
						Target[(i + ii + 2) * TargetSizeOfLine + k] += acc2;
						Target[(i + ii + 3) * TargetSizeOfLine + k] += acc3;

					}
				}
			}

			for (unsigned k = 0; k < BlocksPerBaseVectors; k += ElementsInCacheLine)
				// Doing last run on the not Blocked Base Vectors
			{
				for (unsigned ii = 0; ii < ElementsInCacheLine; ++ii) {

					for (unsigned kk = 0; kk < ElementsInCacheLine; kk += 4) {
						T acc0 = 0;
						T acc1 = 0;
						T acc2 = 0;
						T acc3 = 0;

						for (unsigned j = BlocksPerVector; j < Src1Cols; ++j) {
							T Val = Src2[(i + ii) * Src2SizeOfLine + j];

							acc0 += Src1[j * Src1SizeOfLine + k + kk] * Val;
							acc1 += Src1[j * Src1SizeOfLine + k + kk + 1] * Val;
							acc2 += Src1[j * Src1SizeOfLine + k + kk + 2] * Val;
							acc3 += Src1[j * Src1SizeOfLine + k + kk + 3] * Val;
						}


						Target[(i + ii) * TargetSizeOfLine + k + kk] += acc0;
						Target[(i + ii) * TargetSizeOfLine + k + kk + 1] += acc1;
						Target[(i + ii) * TargetSizeOfLine + k + kk + 2] += acc2;
						Target[(i + ii) * TargetSizeOfLine + k + kk + 3] += acc3;
					}
				}
			}


			for (unsigned ii = 0; ii < ElementsInCacheLine; ++ii) {
				for (unsigned k = BlocksPerBaseVectors; k < Src1Rows; ++k) {
					T acc0 = 0;
					for (unsigned j = BlocksPerVector; j < Src2Rows; ++j) {
						acc0 += Src1[j * Src1SizeOfLine + k] * Src2[(i + ii) * Src2SizeOfLine + j];
					}
					Target[(i + ii) * TargetSizeOfLine + k] += acc0;
				}
			}
		}
	}

	void MultAlgo2_CC_Frame_EE(const unsigned VectorsBlocks, const unsigned BlocksPerVector, const unsigned BlocksPerBaseVectors) {
		for (unsigned j = 0; j < BlocksPerVector; j += ElementsInCacheLine) {
			for (unsigned k = 0; k < BlocksPerBaseVectors; k += ElementsInCacheLine) {
				for (unsigned i = VectorsBlocks; i < Src2Cols; ++i) {
					for (unsigned kk = 0; kk < ElementsInCacheLine; kk += 4) {
						T acc0 = 0;
						T acc1 = 0;
						T acc2 = 0;
						T acc3 = 0;

						for (unsigned jj = 0; jj < ElementsInCacheLine; ++jj) {
							T Val = Src2[i * Src2SizeOfLine + j + jj];

							acc0 += Src1[(j + jj) * Src1SizeOfLine + k + kk] * Val;
							acc1 += Src1[(j + jj) * Src1SizeOfLine + k + kk + 1] * Val;
							acc2 += Src1[(j + jj) * Src1SizeOfLine + k + kk + 2] * Val;
							acc3 += Src1[(j + jj) * Src1SizeOfLine + k + kk + 3] * Val;
						}


						Target[i * TargetSizeOfLine + k + kk] += acc0;
						Target[i * TargetSizeOfLine + k + kk + 1] += acc1;
						Target[i * TargetSizeOfLine + k + kk + 2] += acc2;
						Target[i * TargetSizeOfLine + k + kk + 3] += acc3;

					}
				}
			}
		}
	}

	void MultAlgo2_CC_Frame_EN(const unsigned VectorsBlocks, const unsigned BlocksPerVector, const unsigned BlocksPerBaseVectors) {
		for (unsigned j = 0; j < BlocksPerVector; j += ElementsInCacheLine) {
			for (unsigned k = 0; k < BlocksPerBaseVectors; k += ElementsInCacheLine) {
				for (unsigned i = VectorsBlocks; i < Src2Cols; ++i) {
					for (unsigned kk = 0; kk < ElementsInCacheLine; kk += 4) {
						T acc0 = 0;
						T acc1 = 0;
						T acc2 = 0;
						T acc3 = 0;

						for (unsigned jj = 0; jj < ElementsInCacheLine; ++jj) {
							T Val = Src2[i * Src2SizeOfLine + j + jj];

							acc0 += Src1[(j + jj) * Src1SizeOfLine + k + kk] * Val;
							acc1 += Src1[(j + jj) * Src1SizeOfLine + k + kk + 1] * Val;
							acc2 += Src1[(j + jj) * Src1SizeOfLine + k + kk + 2] * Val;
							acc3 += Src1[(j + jj) * Src1SizeOfLine + k + kk + 3] * Val;
						}

						Target[i * TargetSizeOfLine + k + kk] += acc0;
						Target[i * TargetSizeOfLine + k + kk + 1] += acc1;
						Target[i * TargetSizeOfLine + k + kk + 2] += acc2;
						Target[i * TargetSizeOfLine + k + kk + 3] += acc3;

					}
				}
			}
		}

		for (unsigned k = 0; k < BlocksPerBaseVectors; k += ElementsInCacheLine) {
			for (unsigned i = VectorsBlocks; i < Src2Cols; ++i) {
				for (unsigned kk = 0; kk < ElementsInCacheLine; kk += 4) {
					T acc0 = 0;
					T acc1 = 0;
					T acc2 = 0;
					T acc3 = 0;

					for (unsigned j = BlocksPerVector; j < Src2Rows; ++j) {
						T Val = Src2[i * Src2SizeOfLine + j];

						acc0 += Src1[j * Src1SizeOfLine + k + kk] * Val;
						acc1 += Src1[j * Src1SizeOfLine + k + kk + 1] * Val;
						acc2 += Src1[j * Src1SizeOfLine + k + kk + 2] * Val;
						acc3 += Src1[j * Src1SizeOfLine + k + kk + 3] * Val;
					}

					Target[i * TargetSizeOfLine + k + kk] += acc0;
					Target[i * TargetSizeOfLine + k + kk + 1] += acc1;
					Target[i * TargetSizeOfLine + k + kk + 2] += acc2;
					Target[i * TargetSizeOfLine + k + kk + 3] += acc3;

				}
			}
		}

	}

	void MultAlgo2_CC_Frame_NE(const unsigned VectorsBlocks, const unsigned BlocksPerVector, const unsigned BlocksPerBaseVectors) {
		for (unsigned j = 0; j < BlocksPerVector; j += ElementsInCacheLine) {
			for (unsigned k = 0; k < BlocksPerBaseVectors; k += ElementsInCacheLine) {
				for (unsigned i = VectorsBlocks; i < Src2Cols; ++i) {
					for (unsigned kk = 0; kk < ElementsInCacheLine; kk += 4) {
						T acc0 = 0;
						T acc1 = 0;
						T acc2 = 0;
						T acc3 = 0;

						for (unsigned jj = 0; jj < ElementsInCacheLine; ++jj) {
							T Val = Src2[i * Src2SizeOfLine + (j + jj)];

							acc0 += Src1[(j + jj) * Src1SizeOfLine + k + kk] * Val;
							acc1 += Src1[(j + jj) * Src1SizeOfLine + k + kk + 1] * Val;
							acc2 += Src1[(j + jj) * Src1SizeOfLine + k + kk + 2] * Val;
							acc3 += Src1[(j + jj) * Src1SizeOfLine + k + kk + 3] * Val;
						}

						Target[i * TargetSizeOfLine + k + kk] += acc0;
						Target[i * TargetSizeOfLine + k + kk + 1] += acc1;
						Target[i * TargetSizeOfLine + k + kk + 2] += acc2;
						Target[i * TargetSizeOfLine + k + kk + 3] += acc3;
					}
				}
			}

			for (unsigned i = VectorsBlocks; i < Src2Cols; ++i) {
				for (unsigned k = BlocksPerBaseVectors; k < Src1Rows; ++k) {
					T acc = 0;

					for (unsigned jj = 0; jj < ElementsInCacheLine; ++jj) {
						acc += Src1[(j + jj) * Src1SizeOfLine + k] * Src2[i * Src2SizeOfLine + j + jj];
					}

					Target[i * TargetSizeOfLine + k] += acc;
				}
			}
		}
	}

	void MultAlgo2_CC_Frame_NN(const unsigned VectorsBlocks, const unsigned BlocksPerVector, const unsigned BlocksPerBaseVectors) {
		for (unsigned j = 0; j < BlocksPerVector; j += ElementsInCacheLine) {
			for (unsigned k = 0; k < BlocksPerBaseVectors; k += ElementsInCacheLine) {
				for (unsigned i = VectorsBlocks; i < Src2Cols; ++i) {
					for (unsigned kk = 0; kk < ElementsInCacheLine; kk += 4) {
						T acc0 = 0;
						T acc1 = 0;
						T acc2 = 0;
						T acc3 = 0;

						for (unsigned jj = 0; jj < ElementsInCacheLine; ++jj) {
							T Val = Src2[i * Src2SizeOfLine + j + jj];

							acc0 += Src1[(j + jj) * Src1SizeOfLine + k + kk] * Val;
							acc1 += Src1[(j + jj) * Src1SizeOfLine + k + kk + 1] * Val;
							acc2 += Src1[(j + jj) * Src1SizeOfLine + k + kk + 2] * Val;
							acc3 += Src1[(j + jj) * Src1SizeOfLine + k + kk + 3] * Val;
						}

						Target[i * TargetSizeOfLine + k + kk] += acc0;
						Target[i * TargetSizeOfLine + k + kk + 1] += acc1;
						Target[i * TargetSizeOfLine + k + kk + 2] += acc2;
						Target[i * TargetSizeOfLine + k + kk + 3] += acc3;
					}
				}
			}

			for (unsigned i = VectorsBlocks; i < Src2Cols; ++i) {
				for (unsigned k = BlocksPerBaseVectors; k < Src1Rows; ++k) {
					T acc = 0;

					for (unsigned jj = 0; jj < ElementsInCacheLine; ++jj) {
						acc += Src1[(j + jj) * Src1SizeOfLine + k] * Src2[i * Src2SizeOfLine + j + jj];
					}
					Target[i * TargetSizeOfLine + k] += acc;
				}
			}
		}

		for (unsigned k = 0; k < BlocksPerBaseVectors; k += ElementsInCacheLine) {
			for (unsigned i = VectorsBlocks; i < Src2Cols; ++i) {
				for (unsigned kk = 0; kk < ElementsInCacheLine; kk += 4) {
					T acc0 = 0;
					T acc1 = 0;
					T acc2 = 0;
					T acc3 = 0;

					for (unsigned j = BlocksPerVector; j < Src2Rows; ++j) {
						T Val = Src2[i * Src2SizeOfLine + j];

						acc0 += Src1[j * Src1SizeOfLine + k + kk] * Val;
						acc1 += Src1[j * Src1SizeOfLine + k + kk + 1] * Val;
						acc2 += Src1[j * Src1SizeOfLine + k + kk + 2] * Val;
						acc3 += Src1[j * Src1SizeOfLine + k + kk + 3] * Val;
					}

					Target[i * TargetSizeOfLine + k + kk] += acc0;
					Target[i * TargetSizeOfLine + k + kk + 1] += acc1;
					Target[i * TargetSizeOfLine + k + kk + 2] += acc2;
					Target[i * TargetSizeOfLine + k + kk + 3] += acc3;
				}
			}
		}

		for (unsigned i = VectorsBlocks; i < Src2Cols; ++i) {
			for (unsigned k = BlocksPerBaseVectors; k < Src1Rows; ++k) {
				T acc = 0;

				for (unsigned j = BlocksPerVector; j < Src2Rows; ++j) {
					acc += Src1[j * Src1SizeOfLine + k] * Src2[i * Src2SizeOfLine + j];
				}
				Target[i * TargetSizeOfLine + k] += acc;
			}
		}
	}

	void MultAlgo2_CC()
		// Uses definition of matrix as representation of base vectors and multiplication as function affecting vectors
		// Actualy works ONLY on perfectly blockabled matrices e.g nxn, where n is divisible by ElementsInCacheLine
		// 0.21
	{
		const unsigned VectorsBlocks = (Src2Cols / ElementsInCacheLine) * ElementsInCacheLine;
		const unsigned BlocksPerVector = (Src2Rows / ElementsInCacheLine) * ElementsInCacheLine;
		const unsigned BlocksPerBaseVectors = (Src1Rows / ElementsInCacheLine) * ElementsInCacheLine;


		if (BlocksPerBaseVectors == Src1Rows) {
			if (BlocksPerVector == Src2Rows) {
				MultAlgo2_CC_Blocks_EE(VectorsBlocks, BlocksPerVector, BlocksPerBaseVectors);

				if (VectorsBlocks != Src2Cols) {
					MultAlgo2_CC_Frame_EE(VectorsBlocks, BlocksPerVector, BlocksPerBaseVectors);
				}
			}
			else {
				MultAlgo2_CC_Blocks_EN(VectorsBlocks, BlocksPerVector, BlocksPerBaseVectors);

				if (VectorsBlocks != Src2Cols) {
					MultAlgo2_CC_Frame_EN(VectorsBlocks, BlocksPerVector, BlocksPerBaseVectors);
				}
			}
		}
		else {
			if (BlocksPerVector == Src2Rows) {
				MultAlgo2_CC_Blocks_NE(VectorsBlocks, BlocksPerVector, BlocksPerBaseVectors);

				if (VectorsBlocks != Src2Cols) {
					MultAlgo2_CC_Frame_NE(VectorsBlocks, BlocksPerVector, BlocksPerBaseVectors);
				}
			}
			else {
				MultAlgo2_CC_Blocks_NN(VectorsBlocks, BlocksPerVector, BlocksPerBaseVectors);

				if (VectorsBlocks != Src2Cols) {
					MultAlgo2_CC_Frame_NN(VectorsBlocks, BlocksPerVector, BlocksPerBaseVectors);
				}
			}
		}

	}

	void MultAlgo3_CC()
		// Some kind of abomination
	{
		const unsigned VectorBlockOnTarget = (Src2Cols / ElementsInCacheLine) * ElementsInCacheLine;
		const unsigned BlocksOnBaseVectors = (Src1Rows / ElementsInCacheLine) * ElementsInCacheLine;
		const unsigned BaseVectorBlocks = (Src1Cols / ElementsInCacheLine) * ElementsInCacheLine;

		for (unsigned i = 0; i < VectorBlockOnTarget; i += ElementsInCacheLine) {
			for (unsigned k = 0; k < BlocksOnBaseVectors; k += ElementsInCacheLine) {
				for (unsigned ii = 0; ii < ElementsInCacheLine; ++ii) {
					T Val = Src2[(i + ii) * Src2SizeOfLine];
					for (unsigned kk = 0; kk < ElementsInCacheLine; ++kk) {
						Target[(i + ii) * TargetSizeOfLine + k + kk] = Src1[k + kk] * Val;
					}

					for (unsigned jj = 1; jj < ElementsInCacheLine; ++jj) {
						for (unsigned kk = 0; kk < ElementsInCacheLine; ++kk) {
							Target[(i + ii) * TargetSizeOfLine + k + kk] += Src1[jj * Src1SizeOfLine + k + kk] * Val;
						}
					}
				}
			}

			for (unsigned j = ElementsInCacheLine; j < BaseVectorBlocks; j += ElementsInCacheLine) {
				for (unsigned k = 0; k < BlocksOnBaseVectors; k += ElementsInCacheLine) {
					for (unsigned ii = 0; ii < ElementsInCacheLine; ++ii) {
						for (unsigned jj = 0; jj < ElementsInCacheLine; ++jj) {
							T Val0 = Src2[(i + ii) * Src2SizeOfLine + j + jj];

							for (unsigned kk = 0; kk < ElementsInCacheLine; ++kk) {
								Target[(i + ii) * TargetSizeOfLine + k + kk] += Src1[(j + jj) * Src1SizeOfLine + k + kk] * Val0;
							}
						}
					}
				}
			}
		}
	}
};

template<>
void SimpleMultMachine<double>::MultAlgo2_CC_Blocks_EE(const unsigned VectorsBlocks, const unsigned BlocksPerVector, const unsigned BlocksPerBaseVectors,
	const unsigned VectorsStartingBlock)
// AVX READY version of previous algorithm
// 0.06
{
	for (unsigned i = 0; i < VectorsBlocks; i += ElementsInCacheLine) {
		for (unsigned j = 0; j < BlocksPerVector; j += ElementsInCacheLine)
			// Next iterations without cleaning
		{
			for (unsigned k = 0; k < BlocksPerBaseVectors; k += ElementsInCacheLine) {
				for (unsigned ii = 0; ii < ElementsInCacheLine; ii += 2) {

					for (unsigned kk = 0; kk < ElementsInCacheLine; kk += 4) {
						__m256d acc0 = _mm256_setzero_pd();
						__m256d acc1 = _mm256_setzero_pd();

						for (unsigned jj = 0; jj < ElementsInCacheLine; ++jj) {
							double VectorCoef0 = Src2[(i + ii) * Src2SizeOfLine + j + jj];
							double VectorCoef1 = Src2[(i + ii + 1) * Src2SizeOfLine + j + jj];
							__m256d BaseVectorAVXLine = *((__m256d*)(Src1 + (j + jj) * Src1SizeOfLine + k + kk));

							acc0 = _mm256_fmadd_pd(BaseVectorAVXLine,
								_mm256_set1_pd(VectorCoef0),
								acc0
							);

							acc1 = _mm256_fmadd_pd(BaseVectorAVXLine,
								_mm256_set1_pd(VectorCoef1),
								acc1
							);
						}

						*((__m256d*)(Target + (i + ii) * TargetSizeOfLine + k + kk)) = _mm256_add_pd(acc0,
							*((__m256d*)(Target + (i + ii) * TargetSizeOfLine + k + kk))
						);

						*((__m256d*)(Target + (i + ii + 1) * TargetSizeOfLine + k + kk)) = _mm256_add_pd(acc1,
							*((__m256d*)(Target + (i + ii + 1) * TargetSizeOfLine + k + kk))
						);
					}
				}
			}
		}
	}
}

template<>
void SimpleMultMachine<double>::MultAlgo2_CC_Blocks_EN(const unsigned VectorsBlocks, const unsigned BlocksPerVector, const unsigned BlocksPerBaseVectors,
	const unsigned VectorsStartingBlock)
// AVX READY version of previous algorithm
// 0.06
{
	for (unsigned i = 0; i < VectorsBlocks; i += ElementsInCacheLine) {
		for (unsigned j = 0; j < BlocksPerVector; j += ElementsInCacheLine)
			// Next iterations without cleaning
		{
			for (unsigned k = 0; k < BlocksPerBaseVectors; k += ElementsInCacheLine) {
				for (unsigned ii = 0; ii < ElementsInCacheLine; ii += 2) {

					for (unsigned kk = 0; kk < ElementsInCacheLine; kk += 4) {
						__m256d acc0 = _mm256_set1_pd(0);
						__m256d acc1 = _mm256_set1_pd(0);

						for (unsigned jj = 0; jj < ElementsInCacheLine; ++jj) {
							double VectorCoef0 = Src2[(i + ii) * Src2SizeOfLine + j + jj];
							double VectorCoef1 = Src2[(i + ii + 1) * Src2SizeOfLine + j + jj];
							__m256d BaseVectorAVXLine = *((__m256d*)(Src1 + (j + jj) * Src1SizeOfLine + k + kk));

							acc0 = _mm256_fmadd_pd(BaseVectorAVXLine,
								_mm256_set1_pd(VectorCoef0),
								acc0
							);

							acc1 = _mm256_fmadd_pd(BaseVectorAVXLine,
								_mm256_set1_pd(VectorCoef1),
								acc1
							);
						}

						*((__m256d*)(Target + (i + ii) * TargetSizeOfLine + k + kk)) = _mm256_add_pd(acc0,
							*((__m256d*)(Target + (i + ii) * TargetSizeOfLine + k + kk))
						);

						*((__m256d*)(Target + (i + ii + 1) * TargetSizeOfLine + k + kk)) = _mm256_add_pd(acc1,
							*((__m256d*)(Target + (i + ii + 1) * TargetSizeOfLine + k + kk))
						);
					}
				}
			}
		}

		for (unsigned k = 0; k < BlocksPerBaseVectors; k += ElementsInCacheLine) {
			for (unsigned ii = 0; ii < ElementsInCacheLine; ii += 2) {

				for (unsigned kk = 0; kk < ElementsInCacheLine; kk += 4) {
					__m256d acc0 = _mm256_setzero_pd();
					__m256d acc1 = _mm256_setzero_pd();

					for (unsigned j = BlocksPerVector; j < Src1Cols; ++j) {
						double VectorCoef0 = Src2[(i + ii) * Src2SizeOfLine + j];
						double VectorCoef1 = Src2[(i + ii + 1) * Src2SizeOfLine + j];
						__m256d BaseVectorAVXLine = *((__m256d*)(Src1 + j * Src1SizeOfLine + k + kk));

						acc0 = _mm256_fmadd_pd(BaseVectorAVXLine,
							_mm256_set1_pd(VectorCoef0),
							acc0
						);

						acc1 = _mm256_fmadd_pd(BaseVectorAVXLine,
							_mm256_set1_pd(VectorCoef1),
							acc1
						);
					}

					*((__m256d*)(Target + (i + ii) * TargetSizeOfLine + k + kk)) = _mm256_add_pd(acc0,
						*((__m256d*)(Target + (i + ii) * TargetSizeOfLine + k + kk))
					);

					*((__m256d*)(Target + (i + ii + 1) * TargetSizeOfLine + k + kk)) = _mm256_add_pd(acc1,
						*((__m256d*)(Target + (i + ii + 1) * TargetSizeOfLine + k + kk))
					);
				}
			}
		}
	}
}


template<>
void SimpleMultMachine<double>::MultAlgo2_CC_Blocks_NE(const unsigned VectorsBlocks, const unsigned BlocksPerVector, const unsigned BlocksPerBaseVectors,
	const unsigned VectorsStartingBlock)
// AVX READY version of previous algorithm
// 0.06
{
	for (unsigned i = 0; i < VectorsBlocks; i += ElementsInCacheLine) {
		for (unsigned j = 0; j < BlocksPerVector; j += ElementsInCacheLine)
			// Next iterations without cleaning
		{
			for (unsigned k = 0; k < BlocksPerBaseVectors; k += ElementsInCacheLine) {
				for (unsigned ii = 0; ii < ElementsInCacheLine; ii += 2) {

					for (unsigned kk = 0; kk < ElementsInCacheLine; kk += 4) {
						__m256d acc0 = _mm256_setzero_pd();
						__m256d acc1 = _mm256_setzero_pd();

						for (unsigned jj = 0; jj < ElementsInCacheLine; ++jj) {
							double VectorCoef0 = Src2[(i + ii) * Src2SizeOfLine + j + jj];
							double VectorCoef1 = Src2[(i + ii + 1) * Src2SizeOfLine + j + jj];
							__m256d BaseVectorAVXLine = *((__m256d*)(Src1 + (j + jj) * Src1SizeOfLine + k + kk));

							acc0 = _mm256_fmadd_pd(BaseVectorAVXLine,
								_mm256_set1_pd(VectorCoef0),
								acc0
							);

							acc1 = _mm256_fmadd_pd(BaseVectorAVXLine,
								_mm256_set1_pd(VectorCoef1),
								acc1
							);
						}

						*((__m256d*)(Target + (i + ii) * TargetSizeOfLine + k + kk)) = _mm256_add_pd(acc0,
							*((__m256d*)(Target + (i + ii) * TargetSizeOfLine + k + kk))
						);

						*((__m256d*)(Target + (i + ii + 1) * TargetSizeOfLine + k + kk)) = _mm256_add_pd(acc1,
							*((__m256d*)(Target + (i + ii + 1) * TargetSizeOfLine + k + kk))
						);
					}
				}
			}

			for (unsigned ii = 0; ii < ElementsInCacheLine; ii += 4) {
				for (unsigned k = BlocksPerBaseVectors; k < Src1Rows; ++k) {
					double acc0 = 0;
					double acc1 = 0;
					double acc2 = 0;
					double acc3 = 0;

					for (unsigned jj = 0; jj < ElementsInCacheLine; ++jj) {
						acc0 += Src1[(j + jj) * Src1SizeOfLine + k] * Src2[(i + ii) * Src2SizeOfLine + j + jj];
						acc1 += Src1[(j + jj) * Src1SizeOfLine + k] * Src2[(i + ii + 1) * Src2SizeOfLine + j + jj];
						acc2 += Src1[(j + jj) * Src1SizeOfLine + k] * Src2[(i + ii + 2) * Src2SizeOfLine + j + jj];
						acc3 += Src1[(j + jj) * Src1SizeOfLine + k] * Src2[(i + ii + 3) * Src2SizeOfLine + j + jj];
					}


					Target[(i + ii) * TargetSizeOfLine + k] += acc0;
					Target[(i + ii + 1) * TargetSizeOfLine + k] += acc1;
					Target[(i + ii + 2) * TargetSizeOfLine + k] += acc2;
					Target[(i + ii + 3) * TargetSizeOfLine + k] += acc3;

				}
			}
		}
	}
}

template<>
void SimpleMultMachine<double>::MultAlgo2_CC_Blocks_NN(const unsigned VectorsBlocks, const unsigned BlocksPerVector, const unsigned BlocksPerBaseVectors,
	const unsigned VectorsStartingBlock)
// AVX READY version of previous algorithm
// 0.06
{
	for (unsigned i = 0; i < VectorsBlocks; i += ElementsInCacheLine) {
		for (unsigned j = 0; j < BlocksPerVector; j += ElementsInCacheLine)
			// Next iterations without cleaning
		{
			for (unsigned k = 0; k < BlocksPerBaseVectors; k += ElementsInCacheLine) {
				for (unsigned ii = 0; ii < ElementsInCacheLine; ii += 2) {

					for (unsigned kk = 0; kk < ElementsInCacheLine; kk += 4) {
						__m256d acc0 = _mm256_setzero_pd();
						__m256d acc1 = _mm256_setzero_pd();

						for (unsigned jj = 0; jj < ElementsInCacheLine; ++jj) {
							double VectorCoef0 = Src2[(i + ii) * Src2SizeOfLine + j + jj];
							double VectorCoef1 = Src2[(i + ii + 1) * Src2SizeOfLine + j + jj];
							__m256d BaseVectorAVXLine = *((__m256d*)(Src1 + (j + jj) * Src1SizeOfLine + k + kk));

							acc0 = _mm256_fmadd_pd(BaseVectorAVXLine,
								_mm256_set1_pd(VectorCoef0),
								acc0
							);

							acc1 = _mm256_fmadd_pd(BaseVectorAVXLine,
								_mm256_set1_pd(VectorCoef1),
								acc1
							);
						}

						*((__m256d*)(Target + (i + ii) * TargetSizeOfLine + k + kk)) = _mm256_add_pd(acc0,
							*((__m256d*)(Target + (i + ii) * TargetSizeOfLine + k + kk))
						);

						*((__m256d*)(Target + (i + ii + 1) * TargetSizeOfLine + k + kk)) = _mm256_add_pd(acc1,
							*((__m256d*)(Target + (i + ii + 1) * TargetSizeOfLine + k + kk))
						);
					}
				}
			}

			for (unsigned ii = 0; ii < ElementsInCacheLine; ii += 4) {
				for (unsigned k = BlocksPerBaseVectors; k < Src1Rows; ++k) {
					double acc0 = 0;
					double acc1 = 0;
					double acc2 = 0;
					double acc3 = 0;

					for (unsigned jj = 0; jj < ElementsInCacheLine; ++jj) {
						acc0 += Src1[(j + jj) * Src1SizeOfLine + k] * Src2[(i + ii) * Src2SizeOfLine + j + jj];
						acc1 += Src1[(j + jj) * Src1SizeOfLine + k] * Src2[(i + ii + 1) * Src2SizeOfLine + j + jj];
						acc2 += Src1[(j + jj) * Src1SizeOfLine + k] * Src2[(i + ii + 2) * Src2SizeOfLine + j + jj];
						acc3 += Src1[(j + jj) * Src1SizeOfLine + k] * Src2[(i + ii + 3) * Src2SizeOfLine + j + jj];
					}


					Target[(i + ii) * TargetSizeOfLine + k] += acc0;
					Target[(i + ii + 1) * TargetSizeOfLine + k] += acc1;
					Target[(i + ii + 2) * TargetSizeOfLine + k] += acc2;
					Target[(i + ii + 3) * TargetSizeOfLine + k] += acc3;

				}
			}
		}

		for (unsigned k = 0; k < BlocksPerBaseVectors; k += ElementsInCacheLine) {
			for (unsigned ii = 0; ii < ElementsInCacheLine; ii += 2) {

				for (unsigned kk = 0; kk < ElementsInCacheLine; kk += 4) {
					__m256d acc0 = _mm256_setzero_pd();
					__m256d acc1 = _mm256_setzero_pd();

					for (unsigned j = BlocksPerVector; j < Src1Cols; ++j) {
						double VectorCoef0 = Src2[(i + ii) * Src2SizeOfLine + j];
						double VectorCoef1 = Src2[(i + ii + 1) * Src2SizeOfLine + j];
						__m256d BaseVectorAVXLine = *((__m256d*)(Src1 + j * Src1SizeOfLine + k + kk));

						acc0 = _mm256_fmadd_pd(BaseVectorAVXLine,
							_mm256_set1_pd(VectorCoef0),
							acc0
						);

						acc1 = _mm256_fmadd_pd(BaseVectorAVXLine,
							_mm256_set1_pd(VectorCoef1),
							acc1
						);
					}

					*((__m256d*)(Target + (i + ii) * TargetSizeOfLine + k + kk)) = _mm256_add_pd(acc0,
						*((__m256d*)(Target + (i + ii) * TargetSizeOfLine + k + kk))
					);

					*((__m256d*)(Target + (i + ii + 1) * TargetSizeOfLine + k + kk)) = _mm256_add_pd(acc1,
						*((__m256d*)(Target + (i + ii + 1) * TargetSizeOfLine + k + kk))
					);
				}
			}
		}

		for (unsigned ii = 0; ii < ElementsInCacheLine; ++ii) {
			for (unsigned k = BlocksPerBaseVectors; k < Src1Rows; ++k) {
				double acc0 = 0;
				for (unsigned j = BlocksPerVector; j < Src2Rows; ++j) {
					acc0 += Src1[j * Src1SizeOfLine + k] * Src2[(i + ii) * Src2SizeOfLine + j];
				}
				Target[(i + ii) * TargetSizeOfLine + k] += acc0;
			}
		}
	}
}


template<>
void SimpleMultMachine<double>::MultAlgo1_CC()
{
	const unsigned BlocksPerRun = (Src1Cols / ElementsInCacheLine) * ElementsInCacheLine;
	const unsigned VerticalBlocks = (Src1Rows / ElementsInCacheLine) * ElementsInCacheLine;
	const unsigned HorizontalBlocks = (Src2Cols / ElementsInCacheLine) * ElementsInCacheLine;

	for (unsigned i = 0; i < HorizontalBlocks; i += ElementsInCacheLine) {
		for (unsigned j = 0; j < VerticalBlocks; j += ElementsInCacheLine) {
			//#define SumOverVectors(vector_offset) Src1[(k + kk) * Src1SizeOfLine + j + jj] * Src2[(i + ii + vector_offset) * Src2SizeOfLine + k + kk];
			//#define SaveAccumulatorsOverVector(vector_offset) Target[(i + ii + vector_offset) * TargetSizeOfLine + j + jj]

			for (unsigned k = 0; k < BlocksPerRun; k += ElementsInCacheLine) {

				for (unsigned ii = 0; ii < ElementsInCacheLine; ii += 2) {
					for (unsigned jj = 0; jj < ElementsInCacheLine; jj += 4) {
						__m256d acc[2] = { _mm256_set1_pd(0) };

					}
				}
			}
		}
	}
}

#endif //PARALLELNUM_MATRIX_MULTIPLICATION_H_