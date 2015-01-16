#pragma once

#include "Activity.h"
#include "CudaHelper.h"
#include "Utils.h"

namespace mf {

	template<typename Sorry = int>
	class TGlyphsActivity : public Activity {

	protected:
		VectorFieldInfo m_vfInfo;
		bool m_allocated;
		float m_glyphsSpacing;
		uint2 m_glyphsCount;  // counted by changeSpacing method
		float m_planeX;
		bool m_useLines;
		float m_glyphSize;

		static const uint m_verticesPerGlyph = 9;
		static const uint m_facesPerGlyph = 6;


		GLuint m_verticesVbo;
		cudaGraphicsResource* m_cudaVerticesVboResource;

		GLuint m_normalsVbo;
		cudaGraphicsResource* m_cudaNormalsVboResource;

		GLuint m_indicesVbo;
		cudaGraphicsResource* m_cudaIndicesVboResource;

		GLuint m_colorsVbo;
		cudaGraphicsResource* m_cudaColorsVboResource;

	public:
		TGlyphsActivity(const VectorFieldInfo& vfInfo)
				: m_vfInfo(vfInfo)
				, m_allocated(false)
				, m_glyphsSpacing(0)
				, m_glyphSize(4.0f)
				, m_planeX(0.0f)
				, m_useLines(false)
				, m_verticesVbo(NULL)
				, m_cudaVerticesVboResource(nullptr)
				, m_normalsVbo(NULL)
				, m_cudaNormalsVboResource(nullptr)
				, m_indicesVbo(NULL)
				, m_cudaIndicesVboResource(nullptr)
				, m_colorsVbo(NULL)
				, m_cudaColorsVboResource(nullptr)
		{
			changeSpacing(1.0);
			allocate();
			recompute();
		};


		void changeSpacing(float newSpacingValue) {
			m_glyphsSpacing = newSpacingValue;
			m_glyphsCount.x = (int)(m_vfInfo.realSize.y / m_glyphsSpacing);
			m_glyphsCount.y = (int)(m_vfInfo.realSize.z / m_glyphsSpacing);
			allocate();
		}

		virtual ~TGlyphsActivity() {
			free();
		};


		virtual void recompute() {
			assert(m_allocated);

			auto t1 = std::chrono::high_resolution_clock::now();

			size_t bytesCount;
			float3* d_glyphs;
			checkCudaErrors(cudaGraphicsMapResources(1, &m_cudaVerticesVboResource));
			checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void**)&d_glyphs, &bytesCount, m_cudaVerticesVboResource));

			float3* d_colors;
			checkCudaErrors(cudaGraphicsMapResources(1, &m_cudaColorsVboResource));
			checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void**)&d_colors, &bytesCount, m_cudaColorsVboResource));

			if (m_useLines) {
				runGlyphLinesKernel(m_planeX, m_glyphsCount, make_float2(m_vfInfo.realSize.y, m_vfInfo.realSize.z), m_glyphSize, m_vfInfo.volumeCoordSpaceMult,
					d_glyphs, d_colors);
			}
			else {
				uint3* d_indices;
				checkCudaErrors(cudaGraphicsMapResources(1, &m_cudaIndicesVboResource));
				checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void**)&d_indices, &bytesCount, m_cudaIndicesVboResource));

				float3* d_normals;
				checkCudaErrors(cudaGraphicsMapResources(1, &m_cudaNormalsVboResource));
				checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void**)&d_normals, &bytesCount, m_cudaNormalsVboResource));

				runGlyphArrowsKernel(m_planeX, m_glyphsCount, make_float2(m_vfInfo.realSize.y, m_vfInfo.realSize.z), m_glyphSize, m_vfInfo.volumeCoordSpaceMult,
					d_glyphs, d_indices, d_normals, d_colors);

				checkCudaErrors(cudaGraphicsUnmapResources(1, &m_cudaNormalsVboResource));
				checkCudaErrors(cudaGraphicsUnmapResources(1, &m_cudaIndicesVboResource));
			}

			checkCudaErrors(cudaGraphicsUnmapResources(1, &m_cudaColorsVboResource));
			checkCudaErrors(cudaGraphicsUnmapResources(1, &m_cudaVerticesVboResource));

			auto t2 = std::chrono::high_resolution_clock::now();
			m_lastDuration = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() / 1000.0f;
		}

		virtual void drawControls(float& i, float incI) {

			std::stringstream ss;

			drawString(10, ++i * incI, 0, "Use right mouse for movement");

			ss.str("");
			ss << "[+] [-] Glyphs spacing: " << m_glyphsSpacing;
			drawString(10, ++i * incI, 0, ss.str());

			ss.str("");
			ss << "        that's " << (m_glyphsCount.x * m_glyphsCount.y) << " glyphs";
			drawString(10, ++i * incI, 0, ss.str());

			ss.str("");
			ss << "[q] [w] Glyph size: " << m_glyphSize;
			drawString(10, ++i * incI, 0, ss.str());

			drawString(10, ++i * incI, 0, "  [t]   Toggle primitives");

		}

		virtual bool keyboardCallback(unsigned char key) {
			switch (key) {
				case '+':
					changeSpacing(m_glyphsSpacing * 1.25f);
					break;
				case '-':
					changeSpacing(m_glyphsSpacing / 1.25f);
					break;
				case 'q':
					m_glyphSize *= 1.25f;
					break;
				case 'w':
					m_glyphSize /= 1.25f;
					break;
				case 't':
					m_useLines ^= true;
					break;
				default:
					return false;
			}

			recompute();
			return true;
		}

		virtual bool motionCallback(int /*x*/, int y, int /*dx*/, int /*dy*/, int /*screenWidth*/, int screenHeight, int mouseButtonsState) {
			if (mouseButtonsState == (1 << GLUT_RIGHT_BUTTON)) {
				m_planeX = m_vfInfo.realSize.x - ((float)y / screenHeight) * m_vfInfo.realSize.x;
				recompute();
				return true;
			}

			return false;
		}

		virtual void displayCallback() {
			assert(m_allocated);

			if (m_glyphsCount.x == 0 || m_glyphsCount.y == 0) {
				return;
			}

			glBindBuffer(GL_ARRAY_BUFFER, m_verticesVbo);
			glEnableClientState(GL_VERTEX_ARRAY);
			glVertexPointer(3, GL_FLOAT, 0, NULL);

			glBindBuffer(GL_ARRAY_BUFFER, m_colorsVbo);
			glEnableClientState(GL_COLOR_ARRAY);
			glColorPointer(3, GL_FLOAT, 0, NULL);

			if (!m_useLines) {
				glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_indicesVbo);

				glBindBuffer(GL_ARRAY_BUFFER, m_normalsVbo);
				glEnableClientState(GL_NORMAL_ARRAY);
				glNormalPointer(GL_FLOAT, 0, NULL);
			}


			if (m_useLines) {
				glDrawArrays(GL_LINES, 0, 2 * m_glyphsCount.x * m_glyphsCount.y);
			}
			else {
				glEnable(GL_COLOR_MATERIAL);
				glEnable(GL_LIGHTING);

				glDrawElements(GL_TRIANGLES, 3 * m_facesPerGlyph * m_glyphsCount.x * m_glyphsCount.y, GL_UNSIGNED_INT, NULL);

				glDisable(GL_COLOR_MATERIAL);
				glDisable(GL_LIGHTING);
			}

			glBindBuffer(GL_ARRAY_BUFFER, 0);
			glDisableClientState(GL_NORMAL_ARRAY);
			glDisableClientState(GL_COLOR_ARRAY);
			glDisableClientState(GL_VERTEX_ARRAY);
		};

	protected:

		void allocate() {
			if (m_allocated) {
				free();
			}

			m_allocated = true;

			uint totalGlyphsCount = m_glyphsCount.x * m_glyphsCount.y;
			uint verticesCount = m_useLines ? 2 * totalGlyphsCount : m_verticesPerGlyph * totalGlyphsCount;

			checkCudaErrors(createCudaSharedVbo(&m_verticesVbo, GL_ARRAY_BUFFER, verticesCount * sizeof(float3), &m_cudaVerticesVboResource));
			checkCudaErrors(createCudaSharedVbo(&m_colorsVbo, GL_ARRAY_BUFFER, verticesCount * sizeof(float3), &m_cudaColorsVboResource));

			if (!m_useLines) {
				checkCudaErrors(createCudaSharedVbo(&m_normalsVbo, GL_ARRAY_BUFFER, verticesCount * sizeof(float3), &m_cudaNormalsVboResource));
				uint facesCount = m_facesPerGlyph * totalGlyphsCount;
				checkCudaErrors(createCudaSharedVbo(&m_indicesVbo, GL_ELEMENT_ARRAY_BUFFER, 3 * facesCount * sizeof(uint3), &m_cudaIndicesVboResource));

			}

		}

		void free() {
			if (!m_allocated) {
				return;
			}

			checkCudaErrors(deleteCudaSharedVbo(&m_verticesVbo, m_cudaVerticesVboResource));
			m_verticesVbo = NULL;
			m_cudaVerticesVboResource = nullptr;

			checkCudaErrors(deleteCudaSharedVbo(&m_colorsVbo, m_cudaColorsVboResource));
			m_colorsVbo = NULL;
			m_cudaColorsVboResource = nullptr;

			if (m_cudaNormalsVboResource != nullptr) {
				checkCudaErrors(deleteCudaSharedVbo(&m_normalsVbo, m_cudaNormalsVboResource));
				m_normalsVbo = NULL;
				m_cudaNormalsVboResource = nullptr;
			}

			if (m_cudaIndicesVboResource != nullptr) {
				checkCudaErrors(deleteCudaSharedVbo(&m_indicesVbo, m_cudaIndicesVboResource));
				m_indicesVbo = NULL;
				m_cudaIndicesVboResource = nullptr;
			}

			m_allocated = false;
		}

	};

	typedef TGlyphsActivity<> GlyphsActivity;

}