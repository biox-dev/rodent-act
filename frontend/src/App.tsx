import { ConfigProvider, theme } from 'antd';
import { NavLink, Navigate, Route, Routes } from 'react-router-dom';
import DatasetEditorPage from './pages/DatasetEditorPage';
import DatasetPage from './pages/DatasetPage';
import JobPage from './pages/JobPage';
import JobsPage from './pages/JobsPage';
import NewJobPage from './pages/NewJobPage';

export default function App() {
  return (
    <ConfigProvider
      theme={{
        algorithm: theme.darkAlgorithm,
        token: {
          // Keep antd components on the same palette as the hand-written CSS.
          colorPrimary: '#29d398',
          colorInfo: '#4aa8ff',
          colorBgContainer: '#121821',
          colorBgElevated: '#17202c',
          colorBorder: '#243040',
          colorBorderSecondary: '#243040',
          colorText: '#e6edf3',
          colorTextSecondary: '#8b9bb0',
          borderRadius: 8,
          fontFamily:
            "ui-sans-serif, system-ui, -apple-system, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, 'Noto Sans SC', sans-serif",
        },
      }}
    >
      <div className="app">
      <header className="topbar">
        <div className="brand">
          rodent<span>-act</span>
        </div>
        <nav>
          <NavLink to="/" end>
            新建分析
          </NavLink>
          <NavLink to="/jobs">任务列表</NavLink>
          <NavLink to="/datasets">数据集</NavLink>
        </nav>
        <div className="spacer" />
        <div className="hint">悬尾实验 (TST) · 检测 → 追踪 → 0.5 s 窗口分类</div>
      </header>
      <main>
        <Routes>
          <Route path="/" element={<NewJobPage />} />
          <Route path="/jobs" element={<JobsPage />} />
          <Route path="/jobs/:id" element={<JobPage />} />
          <Route path="/datasets" element={<DatasetPage />} />
          <Route path="/datasets/:name" element={<DatasetEditorPage />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </main>
      </div>
    </ConfigProvider>
  );
}
