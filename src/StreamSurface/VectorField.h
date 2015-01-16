#pragma once
#include "VectorFieldInfo.h"

namespace mf {

	template<typename Sorry = int>
	class TVectorField {

	public:
		float4* data;

		/// step size per axis
		float3 spacings;

		/// dimension of vector field
		uint3 sizes;

		/// start points of axis
		int3 mins;

		float maxMangitude;

	public:
		TVectorField()
			: data(nullptr)
			, spacings(make_float3(0, 0, 0))
			, sizes(make_uint3(0, 0, 0))
			, mins(make_int3(0, 0, 0))
			, maxMangitude(0) {

		}

		virtual ~TVectorField() {
			if (data != nullptr) {
				delete[] data;
				data = nullptr;
			}
		}


		size_t totalSize() const { return (size_t)sizes.x * (size_t)sizes.y * (size_t)sizes.z; }

		float3 realSize() const { return make_float3(sizes.x * spacings.x, sizes.y * spacings.y, sizes.z * spacings.z); }

		void fetchInfo(VectorFieldInfo& vfInfo) {
			vfInfo.cudaDataSize = make_cudaExtent(sizes.x, sizes.y, sizes.z);
			vfInfo.dataSize = sizes;
			vfInfo.realSize = realSize();
			vfInfo.realCoordMin = mins;
			vfInfo.volumeCoordSpaceMult = make_float3(1.0f / spacings.x, 1.0f / spacings.y, 1.0f / spacings.z);
		}

		bool loadFromFile(const std::string& filePath) {

			std::cout << "Loading data from '" << filePath << "'..." << std::endl;

			std::ifstream inputStream(filePath, std::ios::binary);
			assert(inputStream.good());

			bool sizeSet = false;
			spacings = make_float3(1, 1, 1);
			mins = make_int3(0, 0, 0);
			//bool littleEndian = false;

			// reading of header
			for (;;) {
				std::string line;
				std::getline(inputStream, line);
				trim(line);

				if (line.length() == 0) {
					break;  // header ended
				}

				size_t colonPos = line.find(':');
				if (colonPos == std::string::npos) {
					continue;  // unknown non-empty line
				}

				std::string key = line.substr(0, colonPos);  // key is word before colon
				std::string value = line.substr(colonPos + 1);
				trim(key);
				trim(value);

				if (key == "type") {
					if (value != "float") {
						std::cerr << "Data type expected to be 'float', but it is '" << value << "'." << std::endl;
						return false;
					}
					std::cout << "Data type: " << value << std::endl;
				}
				else if (key == "dimension") {
					if (value != "4") {
						std::cerr << "Dimension expected to be '4', but it is '" << value << "'." << std::endl;
						return false;
					}
					std::cout << "Dimension: " << value << std::endl;
				}
				else if (key == "sizes") {
					std::stringstream ss;
					ss << value;
					int size;
					ss >> size;
					ss >> sizes.x;
					ss >> sizes.y;
					ss >> sizes.z;
					if (!ss || !ss.eof()) {
						std::cerr << "Failed to read sizes from string: '" << value << "'." << std::endl;
						return false;
					}
					if (size != 3) {
						std::cerr << "Size of the first component expected to be '3' but it is '" << size << "'." << std::endl;
						return false;
					}
					if (sizes.x < 0 || sizes.y < 0 || sizes.y < 0) {
						std::cerr << "Read sizes are weird: " << sizes.x << ", " << sizes.y << ", " << sizes.z << "'." << std::endl;
						return false;
					}
					std::cout << "Size: " << sizes.x << "x" << sizes.y << "x" << sizes.z << std::endl;
					sizeSet = true;
				}
				else if (key == "spacings") {
					std::stringstream ss;
					ss << value;
					std::string ignoreValue;
					ss >> ignoreValue;  // not interesting value
					ss >> spacings.x;
					ss >> spacings.y;
					ss >> spacings.z;
					if (!ss || !ss.eof()) {
						std::cerr << "Failed to read spacings from string: '" << value << "'." << std::endl;
						return false;
					}
					if (spacings.x < 0 || spacings.y < 0 || spacings.y < 0) {
						std::cerr << "Read spacings are weird: " << spacings.x << ", " << spacings.y << ", " << spacings.z << "'." << std::endl;
						return false;
					}
					std::cout << "Spacings: " << spacings.x << ", " << spacings.y << ", " << spacings.z << std::endl;
				}
				else if (key == "axis mins") {
					std::stringstream ss;
					ss << value;
					std::string ignoreValue;
					ss >> ignoreValue;  // not interesting value
					ss >> mins.x;
					ss >> mins.y;
					ss >> mins.z;
					if (!ss || !ss.eof()) {
						std::cerr << "Failed to read mins from string: '" << value << "'." << std::endl;
						return false;
					}
					std::cout << "Minimums: " << mins.x << ", " << mins.y << ", " << mins.z << std::endl;
				}
				else if (key == "encoding") {
					if (value != "raw") {
						std::cerr << "Encoding expected to be 'raw' but it is '" << value << "'." << std::endl;
						return false;
					}
					std::cout << "Encoding: " << value << std::endl;
				}
				//else if (key == "endian") {
				//	if (value == "little") {
				//		littleEndian = true;
				//	}
				//	else if (value == "big") {
				//		littleEndian = false;
				//	}
				//	else {
				//		std::cerr << "Unknown value of edianess '" << value << "'." << std::endl;
				//		return false;
				//	}
				//	std::cout << "Endian: " << (littleEndian ? "little" : "big") << std::endl;
				//}
				else {
					std::cout << "Ignoring: " << line << std::endl;
				}
			}

			std::cout << "Header information loaded loaded." << std::endl;

			if (!sizeSet) {
				std::cerr << "Size information was not found in the file." << std::endl;
				return false;
			}

			size_t totSize = totalSize();
			size_t sizeInBytes = totSize * sizeof(float4);


			std::cout << "Allocating data (" << (sizeInBytes >> 20) << " MB)" << std::endl;
			data = new float4[totSize];

			std::cout << "Reading binary data..." << std::endl;

			maxMangitude = 0;
			//float3 min = make_float3(1e10f, 1e10f, 1e10f);
			//float3 max = make_float3(-1e10f, -1e10f, -1e10f);

			for (size_t i = 0; i < totSize; ++i) {
				float4* curr = &(data[i]);
				// read x, y, z
				inputStream.read((char*)curr, sizeof(float3));

				//min.x = std::min(min.x, curr->x);
				//min.y = std::min(min.y, curr->y);
				//min.z = std::min(min.z, curr->z);

				//max.x = std::max(max.x, curr->x);
				//max.y = std::max(max.y, curr->y);
				//max.z = std::max(max.z, curr->z);

				// compute w
				curr->w = std::sqrtf(curr->x * curr->x + curr->y * curr->y + curr->z * curr->z);

				maxMangitude = std::max(maxMangitude, curr->w);
			}

			std::cout << "Data successfully loaded." << std::endl;

			//for (size_t i = 0; i < 4; ++i) {
			//	std::cout << data[i].x << ", " << data[i].y << ", " << data[i].z << " (" << data[i].w << ")" << std::endl;
			//}

			//std::cout << "Data min: " << min.x << ", " << min.y << ", " << min.z << std::endl;
			//std::cout << "Data max: " << max.x << ", " << max.y << ", " << max.z << std::endl;

			return true;
		}

	};

	typedef TVectorField<> VectorField;

}