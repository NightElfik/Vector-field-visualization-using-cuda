#include "stdafx.h"
#include "CudaHelper.h"
#include "Utils.h"
#include "VectorField.h"
#include "Activity.h"
#include "VectorFieldInfo.h"

#include "SteamlinesLineActivity.h"
#include "GlyphsActivity.h"


namespace mf {

	// UI window stuff
	int screenWidth = 1024;
	int screenHeight = 768;

	bool wireframe = false;
	bool menuVisible = true;
	bool showWing = false;
	bool showBb = true;

	// mouse controls
	int2 mouseOld;
	int mouseButtons = 0;
	float3 rotate = make_float3(0, 0, 0);
	float2 translate = make_float2(0, 0);
	float zoom = 500;

	// vector field
	VectorFieldInfo vectorFieldInfo;


	Activity* currentActivity;



	extern "C" void initCuda(const float4* h_volume, cudaExtent volumeSize, const std::vector<float4>& vectorMangitudeCtf, float maxVectorMangitude);

	extern "C" void runStreamlinesLineKernel(float3* seeds, uint seedsCount, double dt, uint maxSteps,
			cudaExtent volumeSize, float3 volumeCoordSpaceMult, bool useRk4, uint geometrySampling,
			float3* outputPts, uint* outComputedSteps, float3* outVertexColors);

	extern "C" void runStreamtubesLineKernel(float3* seeds, uint seedsCount, double dt, uint maxSteps,
			cudaExtent volumeSize, float3 volumeCoordSpaceMult, float tubeRadius, bool useRk4, uint geometrySampling,
			float3* outVetrices, uint* outComputedSteps, uint3* outFaces, float3* outVertexNormals, float3* outVertexColors);

	extern "C" void runGlyphLinesKernel(float x, uint2 glyphsCount, float2 worldSize, float glyphLength, float3 volumeCoordSpaceMult,
			float3* outputPts, float3* outVertexColors);
	extern "C" void runGlyphArrowsKernel(float x, uint2 glyphsCount, float2 worldSize, float glyphLength, float3 volumeCoordSpaceMult,
			float3* outVertices, uint3* outFaces, float3* outVertexNormals, float3* outVertexColors);

	extern "C" void runLineAdaptiveExtensionKernel(float maxAllowedLineDist, uint2* linePairs, uint linePairsCount, float3* lineVertices,
			uint verticesPerLine, uint verticesPerSample, uint* lineLengths, float3* seeds, uint2* outLinePairs, uint* outPairsIndex, uint* outLinesIndex, uint linesMaxCount);

	extern "C" void runLineStreamSurfaceKernel(uint2* linePairs, uint linePairsCount, float3* lineVertices, uint verticesPerLine, uint* lineLengths,
			uint3* outFaces, uint* outFacesCounts, float3* outNormals);


	void drawControls() {
		float i = 1;
		float incI = 14;
		std::stringstream ss;

		drawString(10, ++i * incI, 0, "  [i]   Toggle wing");
		drawString(10, ++i * incI, 0, "  [o]   Toggle wireframe");
		drawString(10, ++i * incI, 0, "  [l]   Toggle bounding box");
		drawString(10, ++i * incI, 0, "  [p]   Hide/show this menu");

		drawString(10, ++i * incI, 0, "");

		drawString(10, ++i * incI, 0, "Switch activity to:");
		drawString(10, ++i * incI, 0, "  [1]   Streamlines line");
		drawString(10, ++i * incI, 0, "  [2]   Glyphs plane");

		drawString(10, ++i * incI, 0, "");

		if (currentActivity != nullptr) {
			currentActivity->drawControls(i, incI);
		}

		i = screenHeight / incI - 2.5f;

		if (currentActivity != nullptr) {
			ss.str("");
			ss << "Last process time: " << currentActivity->getLastTimerDuration() << " ms";
			drawString(10, ++i * incI, 0, ss.str());
		}
	}

	void drawVolumeBb() {
		glPushMatrix();

		glScalef(vectorFieldInfo.realSize.x, vectorFieldInfo.realSize.y, vectorFieldInfo.realSize.z);
		glTranslatef(0.5f, 0.5f, 0.5f);

		glColor3f(0.2f, 0.2f, 0.2f);
		glutWireCube(1);

		glPopMatrix();
	}

	void drawWing() {
		glBegin(GL_TRIANGLES);

		glColor3f(0.8f, 0.8f, 0.8f);

		float xMin = (float)-vectorFieldInfo.realCoordMin.x;
		float xMax = (float)-vectorFieldInfo.realCoordMin.x + 238;
		float yMin = (float)-vectorFieldInfo.realCoordMin.y - 145;
		float yMid = (float)-vectorFieldInfo.realCoordMin.y;
		float yMax = (float)-vectorFieldInfo.realCoordMin.y + 145;
		float z = (float)-vectorFieldInfo.realCoordMin.z;
		glNormal3f(0, 0, 1);
		glVertex3f(xMin, yMid, z);
		glVertex3f(xMax, yMin, z);
		glVertex3f(xMax, yMax, z);

		glEnd();
	}

	void displayCallback() {

		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		// set view matrix
		glMatrixMode(GL_MODELVIEW);
		glPushMatrix();
		glTranslatef(zoom, translate.x, translate.y);
		glRotatef(rotate.x, 0.0, 1.0, 0.0);
		glRotatef(rotate.y, 0.0, 0.0, 1.0);
		glTranslatef(vectorFieldInfo.realSize.x / -2.0f, vectorFieldInfo.realSize.y / -2.0f, vectorFieldInfo.realSize.z / -2.0f);


		glPolygonMode(GL_FRONT_AND_BACK, wireframe? GL_LINE : GL_FILL);
		glEnable(GL_DEPTH_TEST);

		if (showBb) {
			drawVolumeBb();
		}
		if (showWing){
			drawWing();
		}

		if (currentActivity != nullptr) {
			currentActivity->displayCallback();
		}

		if (menuVisible) {
			// setup orthogonal projection for text
			glMatrixMode(GL_PROJECTION);
			glPushMatrix();
			glLoadIdentity();
			glOrtho(0, screenWidth, screenHeight - 5, -5, -100, 100);

			glMatrixMode(GL_MODELVIEW);
			glPushMatrix();
			glLoadIdentity();

			glDisable(GL_DEPTH_TEST);

			glColor3f(0.8f, 0.8f, 0.8f);
			drawControls();
			glTranslatef(-1, -1, 0);
			glColor3f(0, 0, 0);
			drawControls();

			glMatrixMode(GL_MODELVIEW);
			glPopMatrix();

			glMatrixMode(GL_PROJECTION);
			glPopMatrix();
		}

		glMatrixMode(GL_MODELVIEW);
		glPopMatrix();
		glutSwapBuffers();
		glutReportErrors();
	}

	void keyboardCallback(unsigned char key, int /*x*/, int /*y*/) {

		bool handled = false;
		if (currentActivity != nullptr) {
			handled = currentActivity->keyboardCallback(key);
		}

		if (!handled) {
			switch (key) {
				case 27:
					exit(EXIT_SUCCESS);
					break;
				case 'p':
					menuVisible ^= true;
					break;
				case 'o':
					wireframe ^= true;
					break;
				case 'i':
					showWing ^= true;
					break;
				case 'l':
					showBb ^= true;
					break;
				case ' ':
					if (currentActivity != nullptr) {
						currentActivity->recompute();
					}
					break;
				case '1':
					if (currentActivity != nullptr) {
						delete currentActivity;
					}
					currentActivity = new StreamlinesLineActivity(vectorFieldInfo);
					break;
				case '2':
					if (currentActivity != nullptr) {
						delete currentActivity;
					}
					currentActivity = new GlyphsActivity(vectorFieldInfo);
					break;
			}
		}

		glutPostRedisplay();
	}

	void mouseCallback(int button, int state, int x, int y) {

		if (button == 3 || button == 4) { // stroll a wheel event
			// Each wheel event reports like a button click, GLUT_DOWN then GLUT_UP
			if (state == GLUT_UP) {
				return; // Disregard redundant GLUT_UP events
			}
			zoom *= (button == 3) ? 1/1.05f : 1.05f;

			glutPostRedisplay();
			return;
		}

		if (state == GLUT_DOWN) {
			mouseButtons |= 1 << button;
		}
		else if (state == GLUT_UP) {
			mouseButtons &= ~(1 << button);
		}

		mouseOld.x = x;
		mouseOld.y = y;

		glutPostRedisplay();
	}

	void motionCallback(int x, int y) {
		int dx = x - mouseOld.x;
		int dy = y - mouseOld.y;

		bool handled = false;
		if (currentActivity != nullptr) {
			handled = currentActivity->motionCallback(x, y, dx, dy, screenWidth, screenHeight, mouseButtons);
		}

		if (!handled) {
			if (mouseButtons == (1 << GLUT_LEFT_BUTTON)) {
				rotate.x -= dy * 0.2f;
				rotate.y += dx * 0.2f;
			}
			else if (mouseButtons == (1 << GLUT_MIDDLE_BUTTON)) {
				translate.x -= dx * 0.2f;
				translate.y -= dy * 0.2f;
			}
		}


		mouseOld.x = x;
		mouseOld.y = y;

		glutPostRedisplay();
	}

	void reshapeCallback(int w, int h) {
		screenWidth = w;
		screenHeight = h;

		glViewport(0, 0, screenWidth, screenHeight);

		glMatrixMode(GL_PROJECTION);
		glLoadIdentity();
		gluPerspective(60.0, (GLdouble)screenWidth / (GLdouble)screenHeight, 1, 10000.0);

		glMatrixMode(GL_MODELVIEW);
		glLoadIdentity();
		gluLookAt(-1, 0, 0, 0, 0, 0, 0, 0, 1);
	}


	bool loadVectorField(int argc, const char* const argv[]) {
		if (argc != 2) {
			std::cerr << "Expected one argument as a file path for data in binary NRRD format." << std::endl;
			return false;
		}


		VectorField vectorField;
		if (!vectorField.loadFromFile(argv[1])) {
			return false;
		}

		vectorField.fetchInfo(vectorFieldInfo);

		std::cout << std::endl;

		std::vector<float4> vectMagnitudeCtf;
		vectMagnitudeCtf.push_back(make_float4(0.4f, 0.6f, 0.9f, 0));
		vectMagnitudeCtf.push_back(make_float4(0, 1, 0, 0));
		vectMagnitudeCtf.push_back(make_float4(0.9f, 0.9f, 0, 0));
		vectMagnitudeCtf.push_back(make_float4(1, 0, 0, 0));

		initCuda(vectorField.data, vectorFieldInfo.cudaDataSize, vectMagnitudeCtf, vectorField.maxMangitude);

		std::cout << "Freeing volume data from computer RAM." << std::endl;
		return true;
	}


	bool initGL(int* argc, char* argv[]) {
		glutInit(argc, argv);  // Create GL context
		glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH | GLUT_MULTISAMPLE);
		glutInitWindowSize(screenWidth, screenHeight);
		glutCreateWindow("CS 530 - Stream surfaces by Marek Fiser");

		std::cout << "Open GL version: "  << (char*)glGetString(GL_VERSION) << std::endl;
		std::cout << "Open GL vendor: "  << (char*)glGetString(GL_VENDOR) << std::endl;

		glewInit();

		if (!glewIsSupported("GL_VERSION_2_0")) {
			std::cerr << "ERROR: Support for necessary OpenGL extensions missing." << std::endl;
			return false;
		}

		std::cout << "Using GLEW Version: " << glewGetString(GLEW_VERSION) << std::endl;

		glEnable(GL_MULTISAMPLE);
		glClearColor(1.0f, 1.0f, 1.0f, 1.0f);

		glShadeModel (GL_SMOOTH);
		glEnable(GL_LINE_SMOOTH);
		glDepthMask(GL_TRUE);
		glColorMaterial(GL_FRONT_AND_BACK, GL_DIFFUSE);

		// good old-fashioned fixed function lighting
		float black[]    = { 0.0f, 0.0f, 0.0f, 1.0f };
		float white[]    = { 1.0f, 1.0f, 1.0f, 1.0f };
		float ambient[]  = { 0.1f, 0.1f, 0.1f, 1.0f };
		float lightPos[] = { 0.0f, 300.0f, 600.0f, 0.0f };

		glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT, ambient);
		//glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, diffuse); -- comes from color
		glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, black);

		glLightfv(GL_LIGHT0, GL_AMBIENT, white);
		//glLightfv(GL_LIGHT0, GL_DIFFUSE, white);
		glLightfv(GL_LIGHT0, GL_SPECULAR, white);
		glLightfv(GL_LIGHT0, GL_POSITION, lightPos);

		glLightModelfv(GL_LIGHT_MODEL_AMBIENT, black);
		//glLightModelfv(GL_LIGHT_MODEL_TWO_SIDE, black);
		//glLightModeli(GL_LIGHT_MODEL_TWO_SIDE, GL_TRUE);

		glEnable(GL_LIGHT0);
		glEnable(GL_NORMALIZE);

		glutReportErrors();

		std::cout << "Open GL initialized." << std::endl << std::endl;
		return true;
	}

	void runGui(int argc, char* argv[]) {

		if (!initGL(&argc, argv)) {
			std::cerr << "Failed to initialize OpenGL." << std::endl;
			exit(EXIT_FAILURE);
		}

		checkCudaErrors(cudaGLSetGLDevice(0));

		// register callbacks
		glutDisplayFunc(displayCallback);
		glutKeyboardFunc(keyboardCallback);
		glutMouseFunc(mouseCallback);
		glutMotionFunc(motionCallback);
		glutReshapeFunc(reshapeCallback);

		//argv[1] = "..\\Data\\tdelta-high.nrrd";
		//argv[1] = "..\\Data\\tdelta-bigsmall.nrrd";
		//argv[1] = "..\\Data\\tdelta-bigbig.nrrd";

		if (!loadVectorField(argc, argv)) {
			std::cerr << "Failed to load input file." << std::endl;
			exit(EXIT_FAILURE);
		}

		std::cout << std::endl << "Application initialized successfully!" << std::endl << std::endl;

		//currentActivity = new StreamlinesLineActivity(vectorFieldInfo);
		//currentActivity = new GlyphsActivity(vectorFieldInfo);

		// start app loop - blocking op
		glutMainLoop();
	}
}


int main(int argc, char* argv[]) {
	mf::runGui(argc, argv);
	return EXIT_SUCCESS;
}

