#pragma once


namespace mf {

	inline void drawString(float x, float y, float z, const std::string& text) {
		glRasterPos3f(x, y, z);
		for (size_t i = 0; i < text.size(); i++) {
			glutBitmapCharacter(GLUT_BITMAP_8_BY_13, text[i]);
		}
	}

	inline void ltrim(std::string& s) {
		s.erase(s.begin(), std::find_if(s.begin(), s.end(), std::not1(std::ptr_fun<int, int>(std::isspace))));
	}

	inline void rtrim(std::string& s) {
		s.erase(std::find_if(s.rbegin(), s.rend(), std::not1(std::ptr_fun<int, int>(std::isspace))).base(), s.end());
	}

	inline void trim(std::string& s) {
		ltrim(s);
		rtrim(s);
	}

}
