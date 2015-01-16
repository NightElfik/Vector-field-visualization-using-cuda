#include "stdafx.h"
#include "CudaHelper.h"
#include "CudaMathHelper.h"

namespace mf {

	texture<float4, cudaTextureType1D, cudaReadModeElementType> vectorMangitudeCtfTex;  // 1D texture for color transfer function.
	texture<float4, cudaTextureType3D, cudaReadModeElementType> vectorFieldTex;  // 3D texture for storing of volumetric vector field.

	cudaArray* d_volumeArray = nullptr;
	cudaArray* d_vectorMangitudeCtfData = nullptr;

	float vectorMangitudeCtfLength;
	float maxMangitude;

	extern "C"
	void initCuda(const float4* h_volume, cudaExtent volumeSize, const std::vector<float4>& vectorMangitudeCtf, float maxVectorMangitude) {

		{
			std::cout << "Initializing vector magnitude color transfer function." << std::endl;
			cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float4>();
			checkCudaErrors(cudaMallocArray(&d_vectorMangitudeCtfData, &channelDesc, vectorMangitudeCtf.size(), 1));
			checkCudaErrors(cudaMemcpyToArray(d_vectorMangitudeCtfData, 0, 0, &vectorMangitudeCtf[0], sizeof(float4) * vectorMangitudeCtf.size(), cudaMemcpyHostToDevice));
			vectorMangitudeCtfTex.normalized = false;
			vectorMangitudeCtfTex.filterMode = cudaFilterModeLinear;
			vectorMangitudeCtfTex.addressMode[0] = cudaAddressModeClamp;
			checkCudaErrors(cudaBindTextureToArray(vectorMangitudeCtfTex, d_vectorMangitudeCtfData, channelDesc));
			vectorMangitudeCtfLength = (float)vectorMangitudeCtf.size();
			maxMangitude = maxVectorMangitude;
		}


		{
			// allocate 3D array
			std::cout << "Allocating CUDA 3D array in device." << std::endl;
			cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float4>();
			checkCudaErrors(cudaMalloc3DArray(&d_volumeArray, &channelDesc, volumeSize));

			// copy data to 3D array
			std::cout << "Copying data to device." << std::endl;
			cudaMemcpy3DParms copyParams = {0};
			copyParams.srcPtr   = make_cudaPitchedPtr((void*)h_volume, volumeSize.width * sizeof(float4), volumeSize.width, volumeSize.height);
			copyParams.dstArray = d_volumeArray;
			copyParams.extent   = volumeSize;
			copyParams.kind     = cudaMemcpyHostToDevice;
			checkCudaErrors(cudaMemcpy3D(&copyParams));

			// set texture parameters
			vectorFieldTex.normalized = false;
			vectorFieldTex.filterMode = cudaFilterModeLinear;  // linear interpolation
			vectorFieldTex.addressMode[0] = cudaAddressModeClamp;
			vectorFieldTex.addressMode[1] = cudaAddressModeClamp;
			vectorFieldTex.addressMode[2] = cudaAddressModeClamp;

			// bind array to 3D texture
			std::cout << "Binding 3D texture." << std::endl;
			checkCudaErrors(cudaBindTextureToArray(vectorFieldTex, d_volumeArray, channelDesc));

			std::cout << "Volume data successfully copied to device." << std::endl;
		}

		checkCudaErrors(cudaDeviceSynchronize());
	}



	__inline__ __device__ float3 findPerpendicular(float3 v) {
		/*float ax = abs(v.x);
		float ay = abs(v.y);
		float az = abs(v.z);

		if (ax >= az && ay >= az) {  // ax, ay are dominant
			return make_float3(-v.y, v.x, 0.0f);
		}
		else if (ax >= ay && az >= ay) {  // ax, az are dominant
			return make_float3(-v.z, 0.0f, v.x);
		}
		else {  // ay, az are dominant
			return make_float3(0.0f, -v.z, v.y);
		}*/
		return make_float3(-v.y, v.x, 0.0f);
	}


	__device__ double4 eulerIntegrate(double3 pos, double dt, float3 volumeCoordSpaceMult) {
		float4 v = tex3D(vectorFieldTex, (float)(pos.x * volumeCoordSpaceMult.x), (float)(pos.y * volumeCoordSpaceMult.y), (float)(pos.z * volumeCoordSpaceMult.z));
		return make_double4(dt * v.x, dt * v.y, dt * v.z, v.w);
	}

	__device__ double4 rk4Integrate(double3 pos, double dt, float3 volumeCoordSpaceMult) {
		float4 k1 = tex3D(vectorFieldTex, (float)(pos.x * volumeCoordSpaceMult.x), (float)(pos.y * volumeCoordSpaceMult.y), (float)(pos.z * volumeCoordSpaceMult.z));
		double dtHalf = dt * 0.5;
		float4 k2 = tex3D(vectorFieldTex, (float)((pos.x + dtHalf * k1.x) * volumeCoordSpaceMult.x), (float)((pos.y + dtHalf * k1.y) * volumeCoordSpaceMult.y), (float)((pos.z + dtHalf * k1.z) * volumeCoordSpaceMult.z));
		float4 k3 = tex3D(vectorFieldTex, (float)((pos.x + dtHalf * k2.x) * volumeCoordSpaceMult.x), (float)((pos.y + dtHalf * k2.y) * volumeCoordSpaceMult.y), (float)((pos.z + dtHalf * k2.z) * volumeCoordSpaceMult.z));
		float4 k4 = tex3D(vectorFieldTex, (float)((pos.x + dt * k3.x) * volumeCoordSpaceMult.x), (float)((pos.y + dt * k3.y) * volumeCoordSpaceMult.y), (float)((pos.z + dt * k3.z) * volumeCoordSpaceMult.z));

		double dtSixth = dt / 6.0;
		return make_double4(
			dtSixth * ((double)k1.x + 2.0 * ((double)k2.x + (double)k3.x) + (double)k4.x),
			dtSixth * ((double)k1.y + 2.0 * ((double)k2.y + (double)k3.y) + (double)k4.y),
			dtSixth * ((double)k1.z + 2.0 * ((double)k2.z + (double)k3.z) + (double)k4.z),
			((double)k1.w + 2.0 * ((double)k2.w + (double)k3.w) + (double)k4.w) / 6.0);
	}

	__global__ void computeStreamlinesLineKernel(float3* seeds, uint seedsCount, double dt, uint maxSteps,
			cudaExtent volumeSize, float3 volumeCoordSpaceMult, bool useRk4, uint geometrySampling, float mangitudeCtfNormalizeMult,
			float3* outputPts, uint* outComputedSteps, float3* outVertexColors) {

		uint id = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
		if (id >= seedsCount) {
			return;
		}

		uint outputPos = id * (maxSteps / geometrySampling + 1);
		double3 position = make_double3(seeds[id].x, seeds[id].y, seeds[id].z);

		//printf("[%i] Pos: %f %f %f\n", id, position.x, position.y, position.z);
		//printf("[%i] World: %i %i %i\n", id, volumeSize.width, volumeSize.height, volumeSize.depth);

		double3 maxWorld;
		maxWorld.x = (double)volumeSize.width / volumeCoordSpaceMult.x;
		maxWorld.y = (double)volumeSize.height / volumeCoordSpaceMult.y;
		maxWorld.z = (double)volumeSize.depth / volumeCoordSpaceMult.z;

		//printf("Start pos: %f %f %f\n", position.x, position.y, position.z);

		outputPts[outputPos].x = (float)position.x;
		outputPts[outputPos].y = (float)position.y;
		outputPts[outputPos].z = (float)position.z;
		++outputPos;

		uint geometryStep = geometrySampling;
		uint step = 1;
		for (; step < maxSteps; ++step) {

			if (position.x < 0 || position.y < 0 || position.z < 0 || position.x > maxWorld.x || position.y > maxWorld.y || position.z > maxWorld.z) {
				//printf("Break at pos: %f %f %f\n", position.x, position.y, position.z);
				break;
			}

			double4 dv = useRk4 ? rk4Integrate(position, dt, volumeCoordSpaceMult) : eulerIntegrate(position, dt, volumeCoordSpaceMult);
			//printf("Vector: %f %f %f\n", dv.x, dv.y, dv.z);
			position.x += dv.x;
			position.y += dv.y;
			position.z += dv.z;

			--geometryStep;
			if (geometryStep == 0) {
				geometryStep = geometrySampling;

				outputPts[outputPos].x = (float)position.x;
				outputPts[outputPos].y = (float)position.y;
				outputPts[outputPos].z = (float)position.z;
				//printf("New pos: %f %f %f\n", position.x, position.y, position.z);

				float4 color = tex1D(vectorMangitudeCtfTex, (float)(dv.w * mangitudeCtfNormalizeMult));
				//printf("Color (%f -> %f): %f %f %f\n", vector.w, vector.w * mangitudeCtfNormalizeMult, color.x, color.y, color.z);
				outVertexColors[outputPos - 1].x = color.x;
				outVertexColors[outputPos - 1].y = color.y;
				outVertexColors[outputPos - 1].z = color.z;
				++outputPos;
			}
		}

		float4 vector = tex3D(vectorFieldTex, (float)(position.x * volumeCoordSpaceMult.x), (float)(position.y * volumeCoordSpaceMult.y), (float)(position.z * volumeCoordSpaceMult.z));

		float4 color = tex1D(vectorMangitudeCtfTex, vector.w  * mangitudeCtfNormalizeMult);
		outVertexColors[outputPos - 1].x = color.x;
		outVertexColors[outputPos - 1].y = color.y;
		outVertexColors[outputPos - 1].z = color.z;


		outComputedSteps[id] = step / geometrySampling;
	}

	extern "C"
	void runStreamlinesLineKernel(float3* seeds, uint seedsCount, double dt, uint maxSteps,
			cudaExtent volumeSize, float3 volumeCoordSpaceMult, bool useRk4, uint geometrySampling,
			float3* outputPts, uint* outComputedSteps, float3* outVertexColors) {

		ushort threadsCount = 32;
		uint requredBlocksCount = (seedsCount + threadsCount - 1) / threadsCount;
		if (requredBlocksCount > 1024) {
			threadsCount = 256;
			requredBlocksCount = (seedsCount + threadsCount - 1) / threadsCount;
		}
		assert(requredBlocksCount < 65536);
		ushort blocksCount = (ushort)requredBlocksCount;

		computeStreamlinesLineKernel<<<blocksCount, threadsCount>>>(seeds, seedsCount, dt, maxSteps, volumeSize, volumeCoordSpaceMult, useRk4, geometrySampling,
				(vectorMangitudeCtfLength / maxMangitude), outputPts, outComputedSteps, outVertexColors);
		checkCudaErrors(cudaDeviceSynchronize());
	}


	__device__ void createTubeBaseVertices(float3 pos, float3 v, float radius, uint baseIndex, float3 color, float3* outVetrices, float3* outNormals, float3* outColors) {
		float3 xAxis = normalize(findPerpendicular(v));
		float3 yAxis = normalize(cross(v, xAxis));

		//printf("vertices %i\n", baseIndex);

		outNormals[baseIndex] = xAxis;
		outVetrices[baseIndex] = pos + xAxis * radius;  // x * cos(0) + y * sin (0)
		outColors[baseIndex] = color;
		++baseIndex;

		v = 0.3090f * xAxis + 0.9511f * yAxis;
		outNormals[baseIndex] = v;
		outVetrices[baseIndex] = pos + v * radius;  // x * cos(72) + y * sin (72)
		outColors[baseIndex] = color;
		++baseIndex;

		v = -0.8090f * xAxis + 0.5878f * yAxis;
		outNormals[baseIndex] = v;
		outVetrices[baseIndex] = pos + v * radius;
		outColors[baseIndex] = color;
		++baseIndex;

		v = -0.8090f * xAxis - 0.5878f * yAxis;
		outNormals[baseIndex] = v;
		outVetrices[baseIndex] = pos + v * radius;  // x * cos(216) + y * sin (216)
		outColors[baseIndex] = color;
		++baseIndex;

		v = 0.3090f * xAxis - 0.9511f * yAxis;
		outNormals[baseIndex] = v;
		outVetrices[baseIndex] = pos + v * radius;  // x * cos(288) + y * sin (288)
		outColors[baseIndex] = color;
	}

	__device__ void createTubeIndices(uint vertexBaseId, uint baseFaceId, uint3* outFaces) {

		//printf("v %i, i %i \n", vertexBaseId, baseFaceId);
		for (uint i = 0; i < 5; ++i) {
			uint iNext = (i + 1) % 5;
			outFaces[baseFaceId++] = make_uint3(vertexBaseId + i, vertexBaseId + iNext, vertexBaseId - 5 + iNext);
			outFaces[baseFaceId++] = make_uint3(vertexBaseId + i, vertexBaseId - 5 + i, vertexBaseId - 5 + iNext);
		}

	}

	__global__ void computeStreamtubesLineKernel(float3* seeds, uint seedsCount, double dt, uint maxSteps,
			cudaExtent volumeSize, float3 volumeCoordSpaceMult, float radius, bool useRk4, uint geometrySampling, float mangitudeCtfNormalizeMult,
			float3* outVetrices, uint* outComputedSteps, uint3* outFaces, float3* outVertexNormals, float3* outVertexColors) {

		uint id = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
		if (id >= seedsCount) {
			return;
		}

		uint outputPos = id * (maxSteps / geometrySampling + 1) * 5;
		//printf("id: %i, maxSteps: %i, outPos: %i \n", id, maxSteps, outputPos);
		double3 position = make_double3(seeds[id].x, seeds[id].y, seeds[id].z);

		float4 vector = tex3D(vectorFieldTex, (float)(position.x * volumeCoordSpaceMult.x), (float)(position.y * volumeCoordSpaceMult.y), (float)(position.z * volumeCoordSpaceMult.z));
		float4 color = tex1D(vectorMangitudeCtfTex, vector.w  * mangitudeCtfNormalizeMult);
		createTubeBaseVertices(make_float3(position.x, position.y, position.z), make_float3(vector.x, vector.y, vector.z), radius, outputPos, make_float3(color.x, color.y, color.z),
			outVetrices, outVertexNormals, outVertexColors);
		outputPos += 5;

		double3 maxWorld;
		maxWorld.x = (double)volumeSize.width / volumeCoordSpaceMult.x;
		maxWorld.y = (double)volumeSize.height / volumeCoordSpaceMult.y;
		maxWorld.z = (double)volumeSize.depth / volumeCoordSpaceMult.z;

		uint geometryStep = geometrySampling;
		uint step = 1;
		for (; step < maxSteps; ++step) {

			if (position.x < 0 || position.y < 0 || position.z < 0 || position.x > maxWorld.x || position.y > maxWorld.y || position.z > maxWorld.z) {
				break;
			}

			double4 dv = useRk4 ? rk4Integrate(position, dt, volumeCoordSpaceMult) : eulerIntegrate(position, dt, volumeCoordSpaceMult);
			position.x += dv.x;
			position.y += dv.y;
			position.z += dv.z;

			--geometryStep;
			if (geometryStep == 0) {
				geometryStep = geometrySampling;

				color = tex1D(vectorMangitudeCtfTex, (float)(dv.w * mangitudeCtfNormalizeMult));
				createTubeBaseVertices(make_float3(position.x, position.y, position.z), make_float3(dv.x, dv.y, dv.z), radius, outputPos, make_float3(color.x, color.y, color.z),
					outVetrices, outVertexNormals, outVertexColors);

				createTubeIndices(outputPos, (outputPos - 5 * id - 5) * 2, outFaces);
				outputPos += 5;
			}
		}

		outComputedSteps[id] = step / geometrySampling;
	}

	extern "C"
	void runStreamtubesLineKernel(float3* seeds, uint seedsCount, double dt, uint maxSteps,
			cudaExtent volumeSize, float3 volumeCoordSpaceMult, float tubeRadius, bool useRk4, uint geometrySampling,
			float3* outVetrices, uint* outComputedSteps, uint3* outFaces, float3* outVertexNormals, float3* outVertexColors) {

		ushort threadsCount = 64;
		uint requredBlocksCount = (seedsCount + threadsCount - 1) / threadsCount;
		if (requredBlocksCount > 1024) {
			threadsCount = 256;
			requredBlocksCount = (seedsCount + threadsCount - 1) / threadsCount;
		}
		assert(requredBlocksCount < 65536);
		ushort blocksCount = (ushort)requredBlocksCount;

		computeStreamtubesLineKernel<<<blocksCount, threadsCount>>>(seeds, seedsCount, dt, maxSteps, volumeSize, volumeCoordSpaceMult, tubeRadius, useRk4, geometrySampling,
				(vectorMangitudeCtfLength / maxMangitude), outVetrices, outComputedSteps, outFaces, outVertexNormals, outVertexColors);
		checkCudaErrors(cudaDeviceSynchronize());
	}

	__global__ void computeGlyphLinesKernel(float x, uint2 glyphsCount, float2 worldSize, float glyphLength, float3 volumeCoordSpaceMult,
			float3* outputPts, float mangitudeCtfNormalizeMult, float3* outVertexColors) {


		uint id = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
		uint totalCount = __umul24(glyphsCount.x, glyphsCount.y);
		if (id >= totalCount) {
			return;
		}

		uint col = id % glyphsCount.x;
		uint row = id / glyphsCount.x;

		float3 position = make_float3(x, col * (worldSize.x / glyphsCount.x), row * (worldSize.y / glyphsCount.y));
		float4 vector = tex3D(vectorFieldTex, position.x * volumeCoordSpaceMult.x, position.y * volumeCoordSpaceMult.y, position.z * volumeCoordSpaceMult.z);

		id *= 2;
		outputPts[id] = position;
		outputPts[id + 1] = position + normalize(make_float3(vector.x, vector.y, vector.z)) * glyphLength * vector.w * mangitudeCtfNormalizeMult * 0.5;

		float4 color = tex1D(vectorMangitudeCtfTex, vector.w * mangitudeCtfNormalizeMult);
		outVertexColors[id].x = color.x;
		outVertexColors[id].y = color.y;
		outVertexColors[id].z = color.z;
		outVertexColors[id + 1].x = color.x;
		outVertexColors[id + 1].y = color.y;
		outVertexColors[id + 1].z = color.z;

	}

	extern "C"
	void runGlyphLinesKernel(float x, uint2 glyphsCount, float2 worldSize, float glyphLength, float3 volumeCoordSpaceMult,
			float3* outputPts, float3* outVertexColors) {

		ushort threadsCount = 256;
		uint requredBlocksCount = (glyphsCount.x * glyphsCount.y + threadsCount - 1) / threadsCount;
		assert(requredBlocksCount < 65536);
		ushort blocksCount = (ushort)requredBlocksCount;

		computeGlyphLinesKernel<<<blocksCount, threadsCount>>>(x, glyphsCount, worldSize, glyphLength, volumeCoordSpaceMult,
				outputPts, (vectorMangitudeCtfLength / maxMangitude), outVertexColors);
		checkCudaErrors(cudaDeviceSynchronize());

	}


	__global__ void computeGlyphArrowsKernel(float x, uint2 glyphsCount, float2 worldSize, float glyphLength, float3 volumeCoordSpaceMult,
			float mangitudeCtfNormalizeMult, float3* outVertices, uint3* outFaces, float3* outVertexNormals, float3* outVertexColors) {


		uint id = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
		uint totalCount = __umul24(glyphsCount.x, glyphsCount.y);
		if (id >= totalCount) {
			return;
		}

		uint col = id % glyphsCount.x;
		uint row = id / glyphsCount.x;

		float3 position = make_float3(x, col * (worldSize.x / glyphsCount.x), row * (worldSize.y / glyphsCount.y));
		float4 vector = tex3D(vectorFieldTex, position.x * volumeCoordSpaceMult.x, position.y * volumeCoordSpaceMult.y, position.z * volumeCoordSpaceMult.z);

		float3 forward = normalize(make_float3(vector.x, vector.y, vector.z));
		float3 xAxis = normalize(findPerpendicular(forward));
		float3 yAxis = normalize(cross(forward, xAxis));


		uint faceId = id * 6;
		uint vertexId = id * 9;

		outFaces[faceId] = make_uint3(vertexId, vertexId + 1, vertexId + 2);
		outFaces[faceId + 1] = make_uint3(vertexId, vertexId + 2, vertexId + 3);
		outFaces[faceId + 2] = make_uint3(vertexId, vertexId + 3, vertexId + 4);
		outFaces[faceId + 3] = make_uint3(vertexId, vertexId + 4, vertexId + 1);
		outFaces[faceId + 4] = make_uint3(vertexId + 5, vertexId + 6, vertexId +7);
		outFaces[faceId + 5] = make_uint3(vertexId + 5, vertexId + 7, vertexId + 8);

		id *= 9;
		outVertexNormals[id] = forward;
		outVertexNormals[id + 1] = xAxis;
		outVertexNormals[id + 2] = yAxis;
		outVertexNormals[id + 3] = -xAxis;
		outVertexNormals[id + 4] = -yAxis;
		forward *= -1;
		outVertexNormals[id + 5] = forward;
		outVertexNormals[id + 6] = forward;
		outVertexNormals[id + 7] = forward;
		outVertexNormals[id + 8] = forward;

		forward *= glyphLength * vector.w * mangitudeCtfNormalizeMult * 0.5;
		xAxis *= glyphLength * 0.1;
		yAxis *= glyphLength * 0.1;

		//printf("Pos: %f %f %f\n", xAxis.x, xAxis.y, xAxis.z);

		outVertices[id] = position - forward;  // forward was multiplied by -1
		outVertices[id + 1] = position + xAxis;
		outVertices[id + 2] = position + yAxis;
		outVertices[id + 3] = position - xAxis;
		outVertices[id + 4] = position - yAxis;
		outVertices[id + 5] = position + xAxis;
		outVertices[id + 6] = position + yAxis;
		outVertices[id + 7] = position - xAxis;
		outVertices[id + 8] = position - yAxis;

		float4 color = tex1D(vectorMangitudeCtfTex, vector.w * mangitudeCtfNormalizeMult);
		float3 color3 = make_float3(color.x, color.y, color.z);
		outVertexColors[id] = color3;
		outVertexColors[id + 1] = color3;
		outVertexColors[id + 2] = color3;
		outVertexColors[id + 3] = color3;
		outVertexColors[id + 4] = color3;
		outVertexColors[id + 5] = color3;
		outVertexColors[id + 6] = color3;
		outVertexColors[id + 7] = color3;
		outVertexColors[id + 8] = color3;

	}

	extern "C"
	void runGlyphArrowsKernel(float x, uint2 glyphsCount, float2 worldSize, float glyphLength, float3 volumeCoordSpaceMult,
			float3* outVertices, uint3* outFaces, float3* outVertexNormals, float3* outVertexColors) {

		ushort threadsCount = 256;
		uint requredBlocksCount = (glyphsCount.x * glyphsCount.y + threadsCount - 1) / threadsCount;
		assert(requredBlocksCount < 65536);
		ushort blocksCount = (ushort)requredBlocksCount;

		computeGlyphArrowsKernel<<<blocksCount, threadsCount>>>(x, glyphsCount, worldSize, glyphLength, volumeCoordSpaceMult, (vectorMangitudeCtfLength / maxMangitude),
				outVertices, outFaces, outVertexNormals, outVertexColors);
		checkCudaErrors(cudaDeviceSynchronize());

	}


	__global__ void computeLineAdaptiveExtensionKernel(float maxAllowedLineDist, uint2* linePairs, uint linePairsCount, float3* lineVertices, uint verticesPerLine, uint verticesPerSample, uint* lineLengths, float3* seeds,
			uint2* outLinePairs, uint* outPairsIndex, uint* outLinesIndex, uint linesMaxCount) {

		uint id = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
		if (id >= linePairsCount) {
			return;
		}

		uint2 currPair = linePairs[id];

		uint outPairIndex = atomicAdd(outPairsIndex, 1);
		// preserve original pair
		outLinePairs[outPairIndex].x = currPair.x;
		outLinePairs[outPairIndex].y = currPair.y;

		uint count = min(lineLengths[currPair.x], lineLengths[currPair.y]);
		if (count < 2) {
			return;
		}

		// test 32 samples
		float maxDist = 0;
		for (uint i = 0; i < count; i += count / 32) {
			float3 v1 = lineVertices[currPair.x * verticesPerLine + i * verticesPerSample];
			float3 v2 = lineVertices[currPair.y * verticesPerLine + i * verticesPerSample];
			maxDist = max(maxDist, length(v1 - v2));
		}

		if (maxDist < maxAllowedLineDist) {
			return;
		}

		uint newLines = max(1.0f, sqrtf(maxDist / maxAllowedLineDist));

		//printf("[%id] extending %i %i to %i new\n", id, currPair.x, currPair.y, newLines);

		if (newLines == 0) {
			return;
		}

		uint outLineIndex = atomicAdd(outLinesIndex, newLines);
		if ((outLineIndex + newLines) >= linesMaxCount) {
			atomicAdd(outLinesIndex, -newLines);
			return;
		}

		float3 seedStep = (seeds[currPair.y] - seeds[currPair.x]) / (newLines + 1);

		uint lastOutPairIndex = outPairIndex;
		for (uint i = 0; i < newLines; ++i) {
			seeds[outLineIndex + i] = seeds[currPair.x] + (i + 1) * seedStep;

			uint outPairIndex = atomicAdd(outPairsIndex, 1);
			outLinePairs[lastOutPairIndex].y = outLineIndex + i;
			outLinePairs[outPairIndex].x = outLineIndex + i;
			outLinePairs[outPairIndex].y = currPair.y;

			lastOutPairIndex = outPairIndex;
		}

	}

	extern "C"
	void runLineAdaptiveExtensionKernel(float maxAllowedLineDist, uint2* linePairs, uint linePairsCount, float3* lineVertices, uint verticesPerLine, uint verticesPerSample, uint* lineLengths, float3* seeds,
			uint2* outLinePairs, uint* outPairsIndex, uint* outLinesIndex, uint linesMaxCount) {

		ushort threadsCount = 32;
		uint requredBlocksCount = (linePairsCount + threadsCount - 1) / threadsCount;
		if (requredBlocksCount > 1024) {
			threadsCount = 256;
			requredBlocksCount = (linePairsCount + threadsCount - 1) / threadsCount;
		}
		assert(requredBlocksCount < 65536);
		ushort blocksCount = (ushort)requredBlocksCount;

		computeLineAdaptiveExtensionKernel<<<blocksCount, threadsCount>>>(maxAllowedLineDist, linePairs, linePairsCount, lineVertices, verticesPerLine, verticesPerSample, lineLengths, seeds,
			outLinePairs, outPairsIndex, outLinesIndex, linesMaxCount);
		checkCudaErrors(cudaDeviceSynchronize());
	}


	__global__ void computeStreamSurfaceKernel(uint2* linePairs, uint linePairsCount, float3* lineVertices, uint verticesPerLine, uint* lineLengths,
			uint3* outFaces, uint* outFacesCounts, float3* outNormals) {

		uint id = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
		if (id >= linePairsCount) {
			//printf("%i: too much\n", id);
			return;
		}

		uint2 currPair = linePairs[id];
		uint2 lengths = make_uint2(lineLengths[currPair.x], lineLengths[currPair.y]);

		if (lengths.x < 2 || lengths.y < 2) {
			outFacesCounts[id] = 0;
			//printf("%i: too much\n", id);
			return;
		}

		//printf("%i: lines %i, %i; lengths: %i, %i\n", id, currPair.x, currPair.y, lengths.x ,lengths.y);

		uint line1Offset = currPair.x * verticesPerLine;
		uint line2Offset = currPair.y * verticesPerLine;

		//printf("[%i] vpl: %i\n", id, verticesPerLine);

		float3* line1 = lineVertices + line1Offset;
		float3* line2 = lineVertices + line2Offset;

		float3* normals1 = outNormals + line1Offset;
		float3* normals2 = outNormals + line2Offset;

		uint maxFaces = verticesPerLine * 2 - 2;
		uint3* faces = outFaces + id * maxFaces;

		uint2 currIndex = make_uint2(0, 0);

		//float totalMaxAllowedLineDist = max(1.0f, 8.0f * max(length(line1[0] - line2[0]), length(line1[0] - line2[1])));
		//float maxAllowedLineDist = 2.0f * length(line1[0] - line2[0]);//length(line1[0] - line2[1]) + length(line2[0] - line1[1]);

		float lastMinDist = length(line1[0] - line2[0]);

		uint oneConnections = 0;
		uint twoConnections = 0;
		uint maxConnections = 8;

		uint faceId;
		for (faceId = 0; faceId < maxFaces; ++faceId) {

			if (currIndex.x + 1 >= lengths.x || currIndex.y + 1 >= lengths.y) {
				break;
			}

			float dist1 = (currIndex.x + 1 < lengths.x) ? length(line1[currIndex.x + 1] - line2[currIndex.y]) : (1.0f / 0.0f);
			float dist2 = (currIndex.y + 1 < lengths.y) ? length(line1[currIndex.x] - line2[currIndex.y + 1]) : (1.0f / 0.0f);

			uint newVertexIndex;
			float3 newVertex;
			uint2 nextIndex;
			float minDist;

			if (dist1 <= dist2) {
				if (oneConnections > maxConnections) {
					break;
				}
				++oneConnections;
				twoConnections = 0;
				minDist = dist1;
				newVertexIndex = line1Offset + currIndex.x + 1;
				newVertex = line1[currIndex.x + 1];
				nextIndex = make_uint2(currIndex.x + 1, currIndex.y);
			}
			else if (dist2 < dist1) {
				if (twoConnections > maxConnections) {
					break;
				}
				++twoConnections;
				oneConnections = 0;
				minDist = dist2;
				newVertexIndex = line2Offset + currIndex.y + 1;
				newVertex = line2[currIndex.y + 1];
				nextIndex = make_uint2(currIndex.x, currIndex.y + 1);
			}

			float lenDirect = length(line1[currIndex.x] - line2[currIndex.y]);
			minDist = min(minDist, lenDirect);
			//if (/*minDist > maxAllowedLineDist || */minDist > totalMaxAllowedLineDist) {
			//	break;
			//}

			float distRatio = minDist / lastMinDist;
			if (distRatio > 1.5) {
				//printf("%i: dist ratio %f\n", id, distRatio);
				break;
			}

			//maxAllowedLineDist = (7.0f * maxAllowedLineDist + 2.0f * minDist) / 8.0f;

			faces[faceId] = make_uint3(line1Offset + currIndex.x, line2Offset + currIndex.y, newVertexIndex);
			//printf("%i: faceId %i [%i, %i, %i] (dist1: %f, dist2: %f)\n", id, faceId, faces[faceId].x, faces[faceId].y, faces[faceId].z, dist1, dist2);
			float3 normal = cross(line1[currIndex.x] - line2[currIndex.y], newVertex - line2[currIndex.y]);
			normal = normalize(normal);

			normals1[currIndex.x] = normal;
			normals2[currIndex.y] = normal;

			currIndex = nextIndex;
			lastMinDist = (3.0f * lastMinDist + minDist) / 4.0f;
		}

		//printf("[%i] faces: %i\n", id, faceId);

		outFacesCounts[id] = faceId;
	}

	extern "C"
	void runLineStreamSurfaceKernel(uint2* linePairs, uint linePairsCount, float3* lineVertices, uint verticesPerLine, uint* lineLengths,
			uint3* outFaces, uint* outFacesCounts, float3* outNormals) {

		ushort threadsCount = 32;
		uint requredBlocksCount = (linePairsCount + threadsCount - 1) / threadsCount;
		if (requredBlocksCount > 1024) {
			threadsCount = 256;
			requredBlocksCount = (linePairsCount + threadsCount - 1) / threadsCount;
		}
		assert(requredBlocksCount < 65536);
		ushort blocksCount = (ushort)requredBlocksCount;

		computeStreamSurfaceKernel<<<blocksCount, threadsCount>>>(linePairs, linePairsCount, lineVertices, verticesPerLine, lineLengths,
			outFaces, outFacesCounts, outNormals);
		checkCudaErrors(cudaDeviceSynchronize());
	}

}