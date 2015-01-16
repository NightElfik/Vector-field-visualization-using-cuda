#pragma once

namespace mf {

	template<typename Sorry = int>
	struct TVectorFieldInfo {

		cudaExtent cudaDataSize;
		uint3 dataSize;

		float3 realSize;
		int3 realCoordMin;

		/// multiplier to transform real coordinates to data coordinates
		float3 volumeCoordSpaceMult;

	};

	typedef TVectorFieldInfo<> VectorFieldInfo;

}