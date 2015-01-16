#pragma once

namespace mf {

	template<typename Sorry = int>
	class TActivity {

	protected:
		float m_lastDuration;

	public:


		float getLastTimerDuration() {
			return m_lastDuration;
		}

		virtual void recompute() = 0;
		virtual void drawControls(float& i, float incI) = 0;
		virtual bool keyboardCallback(unsigned char key) = 0;
		virtual bool motionCallback(int x, int y, int dx, int dy, int screenWidth, int screenHeight, int mouseButtonsState) = 0;
		virtual void displayCallback() = 0;

	};

	typedef TActivity<> Activity;

}