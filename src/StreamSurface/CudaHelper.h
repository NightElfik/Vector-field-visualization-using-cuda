#pragma once

namespace mf {

	// Macro for automatic checking of CUDA operations.
	#define checkCudaErrors(val)    checkCudaResult((val), #val, __FILE__, __LINE__)


	template<typename T>
	void checkCudaResult(T result, const char* const func, const char* const file, int line) {
		if (result) {
			std::stringstream ss;
			ss << "CUDA error at " << file << ":" << line << " code=" << static_cast<uint>(result)
				<< " \"" << func << "\"";
			std::cerr << ss.str() << std::endl;
		}
	}


	inline void createVbo(GLuint* vbo, GLenum target, uint size) {
		// create buffer object
		glGenBuffers(1, vbo);
		glBindBuffer(target, *vbo);

		// initialize buffer
		glBufferData(target, size, 0, GL_DYNAMIC_DRAW);
		glBindBuffer(target, 0);

		glutReportErrors();
	}

	inline cudaError_t createCudaSharedVbo(GLuint* vbo, GLenum target, uint size, cudaGraphicsResource** cudaResource) {
		createVbo(vbo, target, size);

		assert(*cudaResource == nullptr);
		return cudaGraphicsGLRegisterBuffer(cudaResource, *vbo, cudaGraphicsMapFlagsNone);
	}


	inline void deleteVbo(GLuint* vbo) {
		if (*vbo == 0) {
			return;
		}

		glBindBuffer(GL_ARRAY_BUFFER, *vbo);
		glDeleteBuffers(1, vbo);

		*vbo = 0;

		glutReportErrors();
	}

	inline cudaError_t deleteCudaSharedVbo(GLuint* vbo, cudaGraphicsResource* cudaResource) {
		deleteVbo(vbo);

		if (cudaResource == nullptr) {
			return cudaSuccess;
		}
		return cudaGraphicsUnregisterResource(cudaResource);
	}

}